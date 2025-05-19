import os
import stdpopsim
import numpy as np
import tsdate
import zarr
import sys
import tszip
import time as time_
import pandas as pd
import warnings
import click
from tsbrowse import preprocess
import evaluation

#warnings.simplefilter(action="ignore", category=FutureWarning)
#warnings.simplefilter(action="ignore", category=UserWarning)


def simulate(length_multiplier, folder, prefix, tsinfer, model='africa', seed=3, num_individuals=1000, prune=True, **kwargs):
    """
    Simulate genetic data and process using tsinfer.

    \b
    Arguments:
        folder: Base folder for outputs.
        length_multiplier: Multiplier for the contig length.
        prefix: The prefix for output file names.
        num_individuals: Total number of individuals to simulate.
    """
    os.makedirs(folder, exist_ok=True)
    print('Simulating')
    if model == 'africa':
        species = stdpopsim.get_species("HomSap")
        model = species.get_demographic_model("OutOfAfrica_4J17")
        contig = species.get_contig("chr20", mutation_rate=model.mutation_rate, length_multiplier=length_multiplier)
        engine = stdpopsim.get_engine("msprime")
        ceu_count = round(num_individuals * 0.95)
        chb_count = round(num_individuals * 0.01)
        yri_count = num_individuals - ceu_count - chb_count
        samples = {"CEU": ceu_count, "CHB": chb_count, "YRI": yri_count}
        sim_ts = engine.simulate(model, contig, samples, seed=seed, msprime_model='smc_prime')
    elif model == 'bonobo':
        species = stdpopsim.get_species("PanTro")
        model = species.get_demographic_model("BonoboGhost_4K19")
        contig = species.get_contig("chr3", mutation_rate=model.mutation_rate, length_multiplier=length_multiplier)
        bonobo_count = round(num_individuals * 0.3)
        central_count = round(num_individuals * 0.1)
        western_count = num_individuals - bonobo_count - central_count
        samples = {"bonobo": bonobo_count, "central": central_count, "western": western_count}
        engine = stdpopsim.get_engine("msprime")
        sim_ts = engine.simulate(model, contig, samples=samples, seed=seed, msprime_model='smc_prime')

    print('Finished simulation')
    if prune==True:
        sim_ts = evaluation.prune_simulated_ts(sim_ts)
    tszip.compress(sim_ts, os.path.join(folder, f"{prefix}-simulated.trees.tsz"))
    variant_data = tsinfer.SampleData.from_tree_sequence(sim_ts)
    return variant_data

def subset_1kgp(chr, arm, tsinfer, interval=None, **kwargs):
    ds = zarr.open(f'/well/kelleher/users/uuc395/tsinfer-snakemake/data/workflow/zarr_vcfs/chr{chr}/data.zarr')
    combined_mask = ds[f"variant_all_subset_chr{chr}{arm}_region_filterNton23_mask"][:]
    
    if interval is not None:
        left, right = interval
        variant_pos = ds.variant_position[:]
        num_sites_before = np.sum(combined_mask == 0)
        masked_pos = variant_pos[combined_mask == 0]
        min_left, max_right = min(masked_pos), max(masked_pos)
        if left < min_left or right > max_right:
            raise ValueError(f"Requested region [{left}, {right}] is outside of [{min_left}, {max_right}].")
        new_region_mask = ~((variant_pos >= left) & (variant_pos <= right))
        combined_mask = combined_mask | new_region_mask
        num_sites_after = np.sum(combined_mask == 0)
        print(f"Of {num_sites_before} sites, {num_sites_after} are in [{left}, {right}].")

    variant_data = tsinfer.VariantData(ds, 
                                        sample_mask="sample_all_subset_mask",
                                        ancestral_state="variant_ancestral_allele", 
                                        site_mask=combined_mask)
    return variant_data


def time_function(func, version, csv_path, *args, **kwargs):
    """
    Measure the wall time and CPU time of an arbitrary function and append the timing
    results to a CSV file.
    """
    before_wall = time_.perf_counter()
    before_cpu = time_.process_time()
    result = func(*args, **kwargs) 
    wall_time = time_.perf_counter() - before_wall
    cpu_time = time_.process_time() - before_cpu
    step = func.__qualname__
    df = pd.DataFrame({
        'version': [version],
        'step': [step],
        'wall_time': [wall_time],
        'cpu_time': [cpu_time]
    })

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    
    return result

def infer(variant_data, folder, prefix, tsinfer, version='1.0', num_threads=126):    
    os.makedirs(folder, exist_ok=True)
    perf_csv_path = os.path.join(folder, f"{prefix}-{version}-perf.csv")
    if os.path.exists(perf_csv_path):
        os.remove(perf_csv_path)
        
    ancestor_data = time_function(
        func=tsinfer.generate_ancestors,
        version=version,
        csv_path=perf_csv_path,
        variant_data=variant_data,
        progress_monitor=True,
        num_threads=num_threads,
        path=os.path.join(folder, f"{prefix}-{version}-ancestors.zarr")
    )

        # # Store linesweep results in a CSV file
    # matcher = tsinfer.AncestorMatcher(variant_data, ancestor_data)
    # ancestor_grouping = matcher.group_by_linesweep()
    # ancestors_per_epoch = np.zeros(len(ancestor_grouping) + 1)
    # for index, ancestors in ancestor_grouping.items():
    #     ancestors_per_epoch[index] = len(ancestors)
    # df = pd.DataFrame({'ancestors_per_epoch': ancestors_per_epoch.astype(int)})
    # csv_path = os.path.join(folder, f"{prefix}-{version}-linesweep.csv")
    # df.to_csv(csv_path, index=False)
    
    # anc_ts = time_function(
    #     func=tsinfer.match_ancestors,
    #     version=version,
    #     csv_path=perf_csv_path,
    #     variant_data=variant_data,
    #     ancestor_data=ancestor_data,
    #     num_threads=num_threads,
    #     path_compression=False,
    #     progress_monitor=True
    # )
    # tszip.compress(anc_ts, os.path.join(folder, f"{prefix}-{version}-ancestor.trees.tsz"))

    # raw_ts = time_function(
    #     func=tsinfer.match_samples,
    #     version=version,
    #     csv_path=perf_csv_path,
    #     variant_data=variant_data,
    #     ancestors_ts=anc_ts,
    #     num_threads=num_threads,
    #     post_process=False,
    #     path_compression=False,
    #     progress_monitor=True
    # )
    # tszip.compress(raw_ts, os.path.join(folder, f"{prefix}-{version}-raw.trees.tsz"))
    # raw_ts_path_prefix = os.path.join(folder, f"{prefix}-{version}-raw")
    # preprocess.preprocess(raw_ts_path_prefix + '.trees.tsz', raw_ts_path_prefix + '.tsbrowse', show_progress=True)
    
    # ts = tsinfer.post_process(raw_ts)
    # tszip.compress(ts, os.path.join(folder, f"{prefix}-{version}-post-processed.trees.tsz"))
    
    # simp_ts = ts.simplify()
    # sdn_ts = tsdate.util.split_disjoint_nodes(simp_ts)
    # dated_ts = tsdate.date(sdn_ts, mutation_rate=1.29e-08, method='variational_gamma', progress=True)
    # dated_ts_path_prefix = os.path.join(folder, f"{prefix}-{version}-post-processed-dated")
    # tszip.compress(dated_ts, dated_ts_path_prefix + '.trees.tsz')
    # preprocess.preprocess(dated_ts_path_prefix + '.trees.tsz', dated_ts_path_prefix + '.tsbrowse', show_progress=True)


def run(method, folder, prefix, version='1.0', num_threads=64, **kwargs):
    """
    Run a specific method to generate VariantData, followed by inference.
    
    Parameters:
    - method: Name of the method to generate data ('simulate' or 'subset_1kgp')
    - folder: Base folder for outputs
    - prefix: Prefix for output files
    - version: Tsinfer version to use ('1.0' or '0.4')
    - num_threads: Number of threads for inference
    - **kwargs: Additional arguments to pass to the data generation method
    """
    if version == '1.0':
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
    elif version == '0.4':
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer-0.4")
    else:
        raise ValueError("Version must be either '1.0' or '0.4'")
    sys.path.append(tsinfer_path)
    import tsinfer
    
    if method == 'simulate':
        variant_data = simulate(tsinfer=tsinfer, folder=folder, prefix=prefix, **kwargs)
    elif method == 'subset_1kgp':
        variant_data = subset_1kgp(tsinfer=tsinfer, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Available methods: 'simulate', 'subset_1kgp'")
    
    infer(variant_data, folder, prefix, tsinfer=tsinfer, version=version, num_threads=num_threads)
    click.echo(f"Method '{method}' and inference completed successfully.")


@click.command()
@click.argument('method', type=click.Choice(['simulate', 'subset_1kgp']))
@click.option('--folder', default='output', type=str, help='Base folder for outputs.')
@click.option('--prefix', default='run', type=str, help='Prefix for output file names.')
@click.option('--version', default='1.0', type=str, help='Tsinfer version (e.g., "1.0" or "0.4").')
@click.option('--num-threads', default=126, type=int, help='Number of threads for inference.')
# Simulation-specific options
@click.option('--model', default='africa', type=str, help='Simulation model: "africa" or "bonobo".')
@click.option('--num-individuals', default=1000, type=int, help='Number of individuals to simulate.')
@click.option('--length-multiplier', default=1.0, type=float, help='Multiplier for the contig length.')
@click.option('--seed', default=3, type=int, help='Random seed for simulation.')
# 1KGP-specific options
@click.option('--chr', type=str, help='Chromosome for 1KGP subset.')
@click.option('--arm', type=str, help='Chromosome arm for 1KGP subset.')
@click.option('--interval', nargs=2, type=int, help='Genomic interval for 1KGP subset.')
def cli_run(method, folder, prefix, version, num_threads, **kwargs):
    """
    Run a specific method to generate data, followed by inference.
    
    METHOD can be 'simulate' or 'subset_1kgp'.
    """
    run(method, folder, prefix, version, num_threads, **kwargs)


if __name__ == '__main__':
    cli_run()
