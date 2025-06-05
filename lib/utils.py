import numpy as np
import sgkit
import xarray as xr
import pandas as pd
import json
import tskit
import csv
from tqdm import tqdm

def prune_arg(arg):
    mutations_count = np.bincount(arg.mutations_site, minlength=arg.num_sites)
    recurrent = np.where(mutations_count > 1)[0]
    muts_to_remove = []
    for site_id in recurrent:
        muts = arg.site(site_id).mutations
        assert len(muts) > 1
        for mut in muts[1:]:
            muts_to_remove.append(mut.id)
    tables = arg.dump_tables()
    mutations = tables.mutations
    tables.mutations.keep_rows(
        np.isin(range(len(mutations)), muts_to_remove, invert=True)
    )
    return tables.tree_sequence()


def add_zarr_variables(ds, output_path):
    G = ds.call_genotype
    ac = np.sum(G, axis=(1, 2))
    an = G.shape[1]*2 #num_samples*2
    af = ac / an
    assert np.all(af <= 1)
    variables = {
        "variant_allele_count": ac,
        "variant_allele_frequency": af,
        "variant_singleton_mask": ac == 1,
        "variant_ancestral_state": ds.variant_allele[:, 0],
    }
    arrays = {
            name: xr.DataArray(data, dims=["variants"], name=name)
            for name, data in variables.items()
        }
    ds.update(arrays)

    sgkit.save_dataset(
            ds.drop_vars(set(ds.data_vars) - set(arrays.keys())),
            output_path.parent,
            mode="a",
            consolidated=False,
        )
    output_path.touch()

    
def build_ancestor_chunks(anc_data_list, ts, output_dir, chunk_size, metadata_path):
    base_anc_data = anc_data_list[0]
    for anc_data in anc_data_list[1:]:
        assert base_anc_data.num_ancestors == anc_data.num_ancestors
        assert base_anc_data.sequence_length == anc_data.sequence_length
        assert np.array_equal(base_anc_data.sites_position, anc_data.sites_position)
        for sites_1, sites_2 in zip(
            base_anc_data.ancestors_focal_sites, anc_data.ancestors_focal_sites
        ):
            assert np.array_equal(sites_1, sites_2)
            
    records = []
    inf_sites_pos = np.append(base_anc_data.sites_position, base_anc_data.sequence_length)
    ts_sites_pos = np.append(ts.sites_position, ts.sequence_length)
    for inf_node, sites in enumerate(base_anc_data.ancestors_focal_sites):
        for inf_site_id in sites:
            pos = inf_sites_pos[inf_site_id]
            true_site_id = np.searchsorted(ts_sites_pos, pos)
            site = ts.site(true_site_id)
            assert len(site.mutations) == 1
            true_node = site.mutations[0].node
            records.append(
                {
                    "inf_focal_site": inf_site_id,
                    "true_focal_site": true_site_id,
                    "focal_position": pos,
                    "inf_node": inf_node,
                    "true_node": true_node,
                }
            )

    df = pd.DataFrame.from_records(records)
    assert len(df) > 0
    assert not df.isnull().values.any()

    index = df.index.to_numpy()
    index_chunks = [index[i : i + chunk_size] for i in range(0, len(index), chunk_size)]

    for chunk_id, chunk in enumerate(index_chunks):
        chunk_df = df.loc[chunk]
        assert chunk_size >= len(chunk_df) > 0
        chunk_path = output_dir / f"unprocessed-chunk-{chunk_id}.csv"
        chunk_df.to_csv(chunk_path, index=False)

    with open(metadata_path, "w") as meta_file:
        json.dump({"num_chunks": len(index_chunks)}, meta_file)

def build_shared_site_maps(ts, anc_data_map):
    """
    Build arrays of indices into each sites_position array that correspond to shared sites
    across all ancestor data and the true tree sequence.
    """
    shared_pos = ts.sites_position
    last_pos = ts.sites_position[-1]+1

    for version, anc_data in anc_data_map.items():
        shared_pos = np.intersect1d(shared_pos, anc_data.sites_position, assume_unique=True)

    true_shared_idx = np.searchsorted(ts.sites_position, shared_pos)
    anc_shared_idx_maps = {
        v: np.searchsorted(anc.sites_position, shared_pos) for v, anc in anc_data_map.items()
    }
    shared_pos = np.append(shared_pos, last_pos)
    return shared_pos, true_shared_idx, anc_shared_idx_maps

def process_ancestor_chunk(df, ts, anc_data_map, rep, genotype_errors_type, switch_error_rate, mispol_error_rate, output_path):

    if genotype_errors_type == "enabled":
        geno_errors = True
    else:
        geno_errors = False

    shared_pos, true_shared_idx, anc_shared_idx_map = build_shared_site_maps(
        ts, anc_data_map
    )
    true_nodes = np.unique(df.true_node)
    print(f"[INFO] Building expanded TS", flush=True)
    tables = ts.dump_tables()
    flags = tables.nodes.flags
    flags[:] = tskit.NODE_IS_SAMPLE
    tables.nodes.flags = flags
    expanded_ts = tables.tree_sequence()

    print(f"[INFO] Generating genotype matrix", flush=True)
    true_genotypes = expanded_ts.genotype_matrix(samples=true_nodes).T
    assert true_genotypes.shape[0] == len(true_nodes)
    true_index_map = {true_node: i for i, true_node in enumerate(true_nodes)}

    print(f"[INFO] Building dataframe", flush=True)

    with open(output_path, "w", newline="") as f:
        writer = None
        for row in tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="Processing rows",
            ncols=80,
            mininterval=5,
        ):
            true_node = row.true_node
            true_node_index = true_index_map[true_node]
            a = true_genotypes[true_node_index]
            a_shared = a[true_shared_idx]
            segment = np.where(a_shared != tskit.MISSING_DATA)[0]
            assert len(segment) > 0
            true_left = segment[0]
            true_right = segment[-1] + 1
            true_full_haplotype = (a_shared > 0).astype("int8")
            true_time = ts.nodes_time[true_node]
            inf_node = row.inf_node
            true_pos_left = shared_pos[true_left]
            true_pos_right = shared_pos[true_right]

            record = {
                "inferred_node": inf_node,
                "true_node": true_node,
                "replicate": rep,
                "genotype_errors_added": geno_errors,
                "switch_error_rate": switch_error_rate,
                "mispolarisation_error_rate": mispol_error_rate,
                "inf_focal_site": row.inf_focal_site,
                "true_focal_site": row.true_focal_site,
                "focal_position": row.focal_position,
                "true_time": true_time,
                "true_site_left": true_left,
                "true_site_right": true_right,
                "true_site_span": true_right - true_left,
                "true_pos_left": true_pos_left,
                "true_pos_right": true_pos_right,
                "true_pos_span": true_pos_right - true_pos_left,
            }

            olap_left = true_left
            olap_right = true_right
            anc_dict = {}
            anc_interval_dict = {}
            for version, anc_data in anc_data_map.items():
                anc = anc_data.ancestor(inf_node)
                anc_dict[version] = anc
                anc_shared_idx = anc_shared_idx_map[version]
                anc_left  = np.count_nonzero(anc_shared_idx < anc.start)
                anc_right = np.count_nonzero(anc_shared_idx < anc.end)
                assert anc_left < anc_right
                anc_interval_dict[version] = (anc_left, anc_right)
                olap_left = max(olap_left, anc_left)
                olap_right = min(olap_right, anc_right)
            assert olap_left < olap_right
            olap_site_span = olap_right - olap_left
            olap_pos_left = shared_pos[olap_left]
            olap_pos_right = shared_pos[olap_right]
            olap_pos_span = olap_pos_right - olap_pos_left
            true_olap = true_full_haplotype[olap_left:olap_right]

            for version, anc in anc_dict.items():
                anc_shared_idx = anc_shared_idx_map[version]
                inf_left, inf_right = anc_interval_dict[version]
                inf_haplotype = anc.full_haplotype[anc_shared_idx]
                inf_olap = inf_haplotype[olap_left:olap_right]
                assert len(inf_olap) == len(true_olap)
                errors = inf_olap != true_olap
                should_be_0 = true_olap & ~inf_olap
                should_be_1 = ~true_olap & inf_olap
                inf_pos_left = shared_pos[inf_left]
                inf_pos_right = shared_pos[inf_right]
                record.update(
                    {
                        f"inferred_site_left_v{version}": inf_left,
                        f"inferred_site_right_v{version}": inf_right,
                        f"inferred_site_span_v{version}": inf_right - inf_left,
                        f"inferred_site_overshoot_left_v{version}": true_left - inf_left,
                        f"inferred_site_overshoot_right_v{version}": inf_right - true_right,
                        f"inferred_pos_left_v{version}": inf_pos_left,
                        f"inferred_pos_right_v{version}": inf_pos_right,
                        f"inferred_pos_span_v{version}": inf_pos_right - inf_pos_left,
                        f"inferred_pos_overshoot_left_v{version}": true_pos_left
                        - inf_pos_left,
                        f"inferred_pos_overshoot_right_v{version}": inf_pos_right
                        - true_pos_right,
                        f"num_errors_v{version}": np.sum(errors),
                        f"num_should_be_0_v{version}": np.sum(should_be_0),
                        f"num_should_be_1_v{version}": np.sum(should_be_1),
                    }
                )

            record.update(
                {
                    "inferred_time": anc.time,
                    "overlap_site_left": olap_left,
                    "overlap_site_right": olap_right,
                    "overlap_site_span": olap_site_span,
                    "overlap_pos_start": olap_pos_left,
                    "overlap_pos_end": olap_pos_right,
                    "overlap_pos_span": olap_pos_span,
                }
            )

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                writer.writeheader()

            writer.writerow(record)

    print(f"[INFO] Finished writing chunk {output_path}")

