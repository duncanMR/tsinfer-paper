import stdpopsim
import numpy as np
import tskit
import pandas as pd
import os
import sys
from pathlib import Path


def simulate(model_name, contig, samples, length_multiplier, seed):
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model(model_name)
    contig = species.get_contig(contig, mutation_rate=model.mutation_rate, length_multiplier=length_multiplier)
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, contig, samples, seed=seed, msprime_model='smc_prime')
    return ts

def prune_simulated_ts(ts):
    mutations_count = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    recurrent = np.where(mutations_count > 1)[0]
    muts_to_remove = []
    for site_id in recurrent:
        muts = ts.site(site_id).mutations
        assert len(muts) > 1
        for mut in muts[1:]:
            muts_to_remove.append(mut.id)
    tables = ts.dump_tables()
    mutations = tables.mutations
    tables.mutations.keep_rows(np.isin(range(len(mutations)), muts_to_remove, invert=True))
    return tables.tree_sequence()

def remove_singletons(ts):
    sites_to_remove = []
    for variant in ts.variants():
        if np.sum(variant.genotypes > 0) == 1:
            sites_to_remove.append(variant.site.id)
    return ts.delete_sites(sites_to_remove)

def prepare_ancestor_df(anc_data_list, ts, output_dir, chunk_size, done_path):
    base_anc_data = anc_data_list[0]
    for anc_data in anc_data_list[1:]:
        assert base_anc_data.num_ancestors == anc_data.num_ancestors
        assert base_anc_data.sequence_length == anc_data.sequence_length
        assert np.array_equal(base_anc_data.sites_position, anc_data.sites_position)
        for sites_1, sites_2 in zip(base_anc_data.ancestors_focal_sites, anc_data.ancestors_focal_sites):
            assert np.array_equal(sites_1, sites_2)

    assert np.array_equal(base_anc_data.sites_position, ts.sites_position)
    assert base_anc_data.sequence_length == ts.sequence_length

    records = []
    for inf_node, sites in enumerate(base_anc_data.ancestors_focal_sites):
        for site_id in sites:
            site = ts.site(site_id)
            pos = site.position
            assert len(site.mutations) == 1
            true_node = site.mutations[0].node
            records.append({
                "focal_site": site_id,
                "focal_position": pos,
                "inf_node": inf_node,
                "true_node": true_node
            })

    long_df = pd.DataFrame.from_records(records)
    df = long_df.groupby(['inf_node', 'true_node']).agg({
        'focal_site': list,
        'focal_position': list
    }).reset_index()
    df = df.rename(columns={
        'focal_site': 'focal_sites',
        'focal_position': 'focal_positions'
    })
    assert len(df) > 0
    assert not df.isnull().values.any()

    index = df.index.to_numpy()
    index_chunks = [index[i:i + chunk_size] for i in range(0, len(index), chunk_size)]

    with open(done_path, "w") as f_done:
        for i, chunk in enumerate(index_chunks):
            chunk_df = df.loc[chunk]
            chunk_path = output_dir / f"chunk_{i}_unprocessed.csv"
            chunk_df.to_csv(chunk_path, index=False)
            f_done.write(f"chunk_{i}_unprocessed.csv\n")
    done_path.touch()

def process_ancestor_chunk(df, ts, anc_data_dict, output_path):
      df.to_csv(output_path, index=False)
      




def process_ancestor_df(chunk, folder, prefix, versions, tsinfer_path):
            
    sys.path.append(tsinfer_path)
    import tsinfer
    ts = tszip.decompress(os.path.join(folder, f"{prefix}-simulated-pruned.trees.tsz"))
    full_df = pd.read_csv(os.path.join(folder, f"{prefix}-anc_id.csv"))
    subset_df = full_df.loc[chunk]
    anc_data_dict = {}
    for v in versions:
        anc_data_path = os.path.join(folder, f"{prefix}-{v}-ancestors.zarr")
        anc_data_dict[v] = tsinfer.formats.AncestorData.load(anc_data_path)
    true_nodes = np.unique(df.true_node)
    tables = ts.dump_tables()
    flags = tables.nodes.flags
    flags[:] = tskit.NODE_IS_SAMPLE
    tables.nodes.flags = flags
    expanded_ts = tables.tree_sequence()
    true_genotypes = expanded_ts.genotype_matrix(samples=true_nodes).T
    assert true_genotypes.shape[0] == len(true_nodes)
    true_index_map = {}
    for i, true_node in enumerate(true_nodes):
        true_index_map[true_node] = i
    records = []
    sites_position = np.append(ts.sites_position, ts.sequence_length)

    for i, row in subset_df.iterrows():
        true_node = row['true_node']
        true_node_index = true_index_map[true_node]
        a = true_genotypes[true_node_index]
        segment = np.where(a != tskit.MISSING_DATA)[0]
        true_start = segment[0]
        true_end = segment[-1] + 1
        true_full_haplotype = (a > 0).astype("int8")
        true_time = ts.nodes_time[true_node]
        inf_node = row['inf_node']

        record = {
            "inferred_node": inf_node,
            "true_node": true_node,
            "focal_sites": row['focal_sites'],
            "focal_positions": row['focal_positions'],
        }

        anc_dict = {}
        olap_start = true_start
        olap_end = true_end
        for v in versions:
            anc = anc_data_dict[v].ancestor(inf_node)
            anc_dict[v] = anc
            olap_start = max(olap_start, anc.start)
            olap_end = min(olap_end, anc.end)

        olap_site_span = olap_end - olap_start
        olap_pos_start = sites_position[olap_start]
        olap_pos_end = sites_position[olap_end]
        olap_pos_span = olap_pos_end - olap_pos_start
        true_olap = true_full_haplotype[olap_start:olap_end]

        for v in versions:
            anc = anc_dict[v]
            start = anc.start
            end = anc.end
            inf_olap = anc.full_haplotype[olap_start:olap_end]
            errors = inf_olap != true_olap
            should_be_0 = true_olap & ~inf_olap
            should_be_1 = ~true_olap & inf_olap

            start_pos = sites_position[start]
            end_pos = sites_position[end]
            record.update({
                f'inferred_site_left_{v}': start,
                f'inferred_site_right_{v}': end,
                f'inferred_site_span_{v}': end - start,
                f'inferred_pos_left_{v}': start_pos,
                f'inferred_pos_right_{v}': end_pos,
                f'inferred_pos_span_{v}': end_pos - start_pos,
                f'num_errors_{v}': np.sum(errors),
                f'num_should_be_0_{v}': np.sum(should_be_0),
                f'num_should_be_1_{v}': np.sum(should_be_1),
            })

        record.update({
            'inferred_time': anc.time,
            'true_time': true_time,
            'true_site_left': true_start,
            'true_site_right': true_end,
            'true_site_span': true_end - true_start,
            'true_pos_left': sites_position[true_start],
            'true_pos_right': sites_position[true_end],
            'true_pos_span': sites_position[true_end] - sites_position[true_start],
            "overlap_site_left": olap_start,
            "overlap_site_right": olap_end,
            'overlap_site_span': olap_site_span,
            'overlap_pos_start': olap_pos_start,
            'overlap_pos_end': olap_pos_end,
            'overlap_pos_span': olap_pos_span,
        })

        records.append(record)

    df = pd.DataFrame.from_records(records)