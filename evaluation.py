import numpy as np
import tskit
import pandas as pd
import os
import sys

tsinfer_path = os.path.abspath("../tsinfer")
sys.path.append(tsinfer_path)
import tsinfer

def prune_simulated_ts(ts):
    sites_to_remove = []
    muts_to_remove = []
    for variant in ts.variants():
        num_alleles = len(variant.alleles) - int(variant.alleles[-1] is None)
        counts = tsinfer.allele_counts(variant.genotypes)
        site = variant.site
        assert site.ancestral_state is not None
        muts = site.mutations
        num_removed_muts = 0
        if len(muts) > 1:
            if num_alleles > 2: #multi-allelic
                sites_to_remove.append(site.id)
            else: #recurrent
                num_removed_muts = len(muts)-1
                for mut in muts[1:]: #delete all but the first mutation
                    muts_to_remove.append(mut.id)

        derived_count = counts.derived - num_removed_muts
        if (derived_count == 1 or derived_count == counts.known):
             sites_to_remove.append(site.id)

    print(f'Removing {len(muts_to_remove)} mutations')
    tables = ts.dump_tables()
    mutations = tables.mutations
    tables.mutations.keep_rows(np.isin(range(len(mutations)), muts_to_remove, invert=True))
    mod_ts = tables.tree_sequence()
    print(mod_ts.num_sites)
    final_ts = mod_ts.delete_sites(sites_to_remove)
    return final_ts

def build_evaluation_df(sim_ts, new_anc_data, old_anc_data):
    ts = prune_simulated_ts(sim_ts)
    assert new_anc_data.num_ancestors == old_anc_data.num_ancestors
    assert np.array_equal(new_anc_data.sites_position, old_anc_data.sites_position)

    seq_length = ts.sequence_length
    nodes_time = ts.nodes_time
    assert np.array_equal(ts.sites_position, new_anc_data.sites_position)
    assert seq_length == new_anc_data.sequence_length
    sites_position = np.append(ts.sites_position, ts.sequence_length)
    focal_sites = []
    site_to_inf_anc = {}
    nodes_set = set()
    true_anc_to_site = {}
    for anc_id, sites in enumerate(new_anc_data.ancestors_focal_sites):
        for site_id in sites:
            focal_sites.append(site_id)
            site_to_inf_anc[site_id] = anc_id
            site = ts.site(site_id)
            assert len(site.mutations) == 1 
            node_id = site.mutations[0].node
            true_anc_to_site[node_id] = site_id
            nodes_set.add(node_id)
    
    true_nodes = np.array(list(nodes_set))
    true_times = nodes_time[true_nodes]
    true_nodes = true_nodes[np.argsort(-true_times)]

    tables = ts.dump_tables()
    flags = tables.nodes.flags
    flags[:] = tskit.NODE_IS_SAMPLE
    tables.nodes.flags = flags
    expanded_ts = tables.tree_sequence()
    ancestors_time = ts.nodes_time[true_nodes]
    A = expanded_ts.genotype_matrix(samples=true_nodes).T
    assert len(A) == len(true_nodes)

    data = []
    for i, a in enumerate(A):
        true_anc_id = true_nodes[i]
        true_time = ancestors_time[i]
        focal_site = true_anc_to_site[true_anc_id]
        inf_anc_id = site_to_inf_anc[focal_site]
        focal_pos = sites_position[focal_site]
        segment = np.where(a != tskit.MISSING_DATA)[0]
        assert len(segment) > 0
        true_start = segment[0]
        true_end = segment[-1] + 1
        assert np.all(a[true_start:true_end] != tskit.MISSING_DATA)
        assert np.all(a[:true_start] == tskit.MISSING_DATA)
        assert np.all(a[true_end:] == tskit.MISSING_DATA)
        true_full_haplotype = (a > 0).astype("int8")

        new_inf_anc = new_anc_data.ancestor(inf_anc_id)
        old_inf_anc = old_anc_data.ancestor(inf_anc_id)
        new_inf_full_haplotype = new_inf_anc.full_haplotype
        old_inf_full_haplotype = old_inf_anc.full_haplotype
        olap_start = max(true_start, new_inf_anc.start, old_inf_anc.start)
        olap_end = min(true_end, new_inf_anc.end, old_inf_anc.end)
        true_olap = true_full_haplotype[olap_start:olap_end]
        new_inf_olap = new_inf_full_haplotype[olap_start:olap_end]
        old_inf_olap = old_inf_full_haplotype[olap_start:olap_end]
        olap_site_span = olap_end - olap_start
        new_errors = new_inf_olap != true_olap
        new_should_be_0 = true_olap & ~new_inf_olap
        new_should_be_1 = ~true_olap & new_inf_olap
        old_errors = old_inf_olap != true_olap
        old_should_be_0 = true_olap & ~old_inf_olap
        old_should_be_1 = ~true_olap & old_inf_olap
        
        # Extract inferred ancestor data
        row = {
            "true_index": true_anc_id,
            "inferred_index": inf_anc_id,
            "true_time": true_time,
            "inferred_time": new_inf_anc.time,
            "focal_site": focal_site,
            "focal_pos": focal_pos,
            "num_errors_new": np.sum(new_errors),
            "num_should_be_0_new": np.sum(new_should_be_0),
            "num_should_be_1_new": np.sum(new_should_be_1),
            "num_errors_old": np.sum(old_errors),
            "num_should_be_0_old": np.sum(old_should_be_0),
            "num_should_be_1_old": np.sum(old_should_be_1),
            "inferred_new_site_left": new_inf_anc.start,
            "inferred_new_site_right": new_inf_anc.end,
            "inferred_new_site_span": new_inf_anc.end - new_inf_anc.start,
            "inferred_new_pos_left": sites_position[new_inf_anc.start],
            "inferred_new_pos_right": sites_position[new_inf_anc.end],
            "inferred_new_pos_span": sites_position[new_inf_anc.end] - sites_position[new_inf_anc.start],
            "inferred_old_site_left": old_inf_anc.start,
            "inferred_old_site_right": old_inf_anc.end,
            "inferred_old_site_span": old_inf_anc.end - old_inf_anc.start,
            "inferred_old_pos_left": sites_position[old_inf_anc.start],
            "inferred_old_pos_right": sites_position[old_inf_anc.end],
            "inferred_old_pos_span": sites_position[old_inf_anc.end] - sites_position[old_inf_anc.start],
            "true_site_left": true_start,
            "true_site_right": true_end,
            "true_site_span": true_end - true_start,
            "true_pos_left": sites_position[true_start],
            "true_pos_right": sites_position[true_end],
            "true_pos_span": sites_position[true_end] - sites_position[true_start],
            "overlap_site_span": olap_site_span,
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df