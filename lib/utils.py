import numpy as np
import sgkit
import xarray as xr
import pandas as pd
import json
import tskit
import csv
from tqdm import tqdm


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

    assert np.array_equal(base_anc_data.sites_position, ts.sites_position)

    records = []
    for inf_node, sites in enumerate(base_anc_data.ancestors_focal_sites):
        for site_id in sites:
            site = ts.site(site_id)
            pos = site.position
            assert len(site.mutations) == 1
            true_node = site.mutations[0].node
            records.append(
                {
                    "focal_site": site_id,
                    "focal_position": pos,
                    "inf_node": inf_node,
                    "true_node": true_node,
                }
            )

    long_df = pd.DataFrame.from_records(records)
    df = (
        long_df.groupby(["inf_node", "true_node"])
        .agg({"focal_site": list, "focal_position": list})
        .reset_index()
    )
    df = df.rename(
        columns={"focal_site": "focal_sites", "focal_position": "focal_positions"}
    )
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


def process_ancestor_chunk(df, ts, sites_position, anc_data_dict, rep, output_path):
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
            segment = np.where(a != tskit.MISSING_DATA)[0]
            true_left = segment[0]
            true_right = segment[-1] + 1
            true_full_haplotype = (a > 0).astype("int8")
            true_time = ts.nodes_time[true_node]
            inf_node = row.inf_node
            true_pos_left = sites_position[true_left]
            true_pos_right = sites_position[true_right]

            record = {
                "inferred_node": inf_node,
                "true_node": true_node,
                "replicate": rep,
                "focal_sites": row.focal_sites,
                "focal_positions": row.focal_positions,
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
            for version, anc_data in anc_data_dict.items():
                anc = anc_data.ancestor(inf_node)
                anc_dict[version] = anc
                olap_left = max(olap_left, anc.start)
                olap_right = min(olap_right, anc.end)

            olap_site_span = olap_right - olap_left
            olap_pos_left = sites_position[olap_left]
            olap_pos_right = sites_position[olap_right]
            olap_pos_span = olap_pos_right - olap_pos_left
            true_olap = true_full_haplotype[olap_left:olap_right]

            for version, anc in anc_dict.items():
                inf_left = anc.start
                inf_right = anc.end
                inf_olap = anc.full_haplotype[olap_left:olap_right]
                errors = inf_olap != true_olap
                should_be_0 = true_olap & ~inf_olap
                should_be_1 = ~true_olap & inf_olap
                inf_pos_left = sites_position[inf_left]
                inf_pos_right = sites_position[inf_right]
                record.update(
                    {
                        f"inferred_site_left_{version}": inf_left,
                        f"inferred_site_right_{version}": inf_right,
                        f"inferred_site_span_{version}": inf_right - inf_left,
                        f"inferred_overshoot_left_{version}": true_left - inf_left,
                        f"inferred_overshoot_right_{version}": inf_right - true_right,
                        f"inferred_pos_left_{version}": inf_pos_left,
                        f"inferred_pos_right_{version}": inf_pos_right,
                        f"inferred_pos_span_{version}": inf_pos_right - inf_pos_left,
                        f"inferred_pos_overshoot_left_{version}": true_pos_left
                        - inf_pos_left,
                        f"inferred_pos_overshoot_right_{version}": inf_pos_right
                        - true_pos_right,
                        f"num_errors_{version}": np.sum(errors),
                        f"num_should_be_0_{version}": np.sum(should_be_0),
                        f"num_should_be_1_{version}": np.sum(should_be_1),
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

def remove_singletons(arg):
    sites_to_remove = []
    for variant in arg.variants():
        if np.sum(variant.genotypes > 0) == 1:
            sites_to_remove.append(variant.site.id)
    return arg.delete_sites(sites_to_remove)

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