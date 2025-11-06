import json
import numpy as np
import pandas as pd
import msprime
import concurrent.futures
import tscompare
import click
import os
import sys

def generate_args(
    num_samples,
    sequence_length,
    ancestry_seed=1,
    mutation_seed=2,
    progress_monitor=None,
    num_threads=190,
):
    base_ts = msprime.sim_ancestry(
        samples=int(num_samples),
        sequence_length=sequence_length,
        recombination_rate=1.29e-8,
        population_size=1e4,
        random_seed=ancestry_seed,
    )
    true_ts = msprime.sim_mutations(base_ts, rate=1.29e-8, random_seed=mutation_seed)
    sample_data = tsinfer.SampleData.from_tree_sequence(true_ts)
    anc = tsinfer.generate_ancestors(sample_data, progress_monitor=progress_monitor, num_threads=num_threads)
    ancestors_ts = tsinfer.match_ancestors(
        sample_data, anc, progress_monitor=progress_monitor, num_threads=num_threads
    )
    raw_ts = tsinfer.match_samples(
        sample_data, ancestors_ts, post_process=False, progress_monitor=progress_monitor, num_threads=num_threads
    )
    inf_ts = tsinfer.post_process(raw_ts)
    return true_ts, inf_ts


def calculate_metrics(true_ts, inf_ts):
    arf = tscompare.haplotype_arf(true_ts, inf_ts).arf
    true_ts = true_ts.simplify()
    inf_ts = inf_ts.simplify()
    L = true_ts.sequence_length
    n = inf_ts.num_samples
    kc_sum = 0.0
    rf_sum = 0.0
    for interval, inf_tree, true_tree in inf_ts.coiterate(true_ts, sample_lists=True):
        if inf_tree.num_edges > 0 and true_tree.num_edges > 0:
            length = interval.right - interval.left
            split_tree = inf_tree.split_polytomies(
                method="random", sample_lists=True
            )
            kc_sum += split_tree.kc_distance(true_tree) * length
            rf_sum += split_tree.rf_distance(true_tree) * length
    kc = kc_sum / L
    rf = rf_sum / (L * (2 * n - 4))
    return kc, rf, arf


def _worker(args):
    ns, rep, ancestry_seed, mutation_seed, sequence_length, num_threads = args
    true_ts, inf_ts = generate_args(ns, sequence_length, ancestry_seed, mutation_seed, num_threads=num_threads)
    kc, rf, arf = calculate_metrics(true_ts, inf_ts)
    return {
        "num_samples": ns,
        "replicate": rep,
        "kc_dist": kc,
        "rf_dist": rf,
        "arf_dist": arf,
        "num_mutations": true_ts.num_mutations,
        "num_true_edges": true_ts.num_edges,
    }


def run_metric_benchmark(
    num_samples_list,
    num_replicates,
    sequence_length=1e5,
    version="0.5",
    random_seed=2,
    csv_path="data/metrics.csv",
    max_workers=4,
    num_threads=47,
):
    rows = []
    rng = np.random.RandomState(random_seed)
    tasks = []
    for ns in num_samples_list:
        seeds = rng.randint(1, 2**31, size=(num_replicates, 2))
        for rep in range(num_replicates):
            aseed, mseed = seeds[rep]
            tasks.append((int(ns), int(rep), int(aseed), int(mseed), sequence_length, int(num_threads)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_worker, tasks, chunksize=1):
            rows.append(res)

    df = pd.DataFrame(rows)
    df["sequence_length"] = sequence_length
    df["version"] = version
    df.to_csv(csv_path, index=False)


def extract_matching_time(ts, commands):
    matches = []
    for prov in ts.provenances():
        record = json.loads(prov.record)
        command = record.get("parameters", {}).get("command")
        if command in commands:
            resources = record.get("resources", {})
            matches.append({**resources, "command": command})
    df = pd.DataFrame(matches)
    return df


def run_perf_benchmark(
    num_samples_list,
    sequence_length,
    version="0.5",
    csv_path="data/perf.csv",
    ancestry_seed=1,
    mutation_seed=2,
):
    df_list = []
    for num_samples in num_samples_list:
        _, inf_ts = generate_args(
            num_samples,
            sequence_length,
            ancestry_seed,
            mutation_seed,
            progress_monitor=True,
        )
        df = extract_matching_time(inf_ts, ["match_ancestors", "match_samples"])
        df["num_samples"] = int(num_samples)
        df["num_mutations"] = inf_ts.num_mutations
        df["num_edges"] = inf_ts.num_edges
        df_list.append(df)
    comb_df = pd.concat(df_list)
    comb_df["version"] = version
    comb_df["sequence_length"] = sequence_length
    comb_df.to_csv(csv_path, index=False)


@click.command()
@click.option(
    "--type",
    "bench_type",
    type=click.Choice(["metric", "perf"], case_sensitive=False),
    required=True,
    help="Benchmark type to run.",
)
@click.option(
    "--num-samples",
    multiple=True,
    required=True,
    type=str,
    help="Sample sizes. Repeat flag or use comma/space list, e.g. "
         '--num-samples 100 --num-samples 200 or --num-samples "100,200 1e3".',
)
@click.option(
    "--sequence-length",
    type=float,
    default=1e5,
    show_default=True,
    help="Sequence length for simulations.",
)
@click.option(
    "--version",
    type=str,
    default="0.5",
    show_default=True,
    help="Method/version label recorded in output CSV.",
)
@click.option(
    "--csv-path",
    type=str,
    required=True,
    help="Output CSV path.",
)
@click.option(
    "--num-replicates",
    type=int,
    default=100,
    show_default=True,
    help="Number of replicates (metric benchmark only).",
)
@click.option(
    "--random-seed",
    type=int,
    default=2,
    show_default=True,
    help="Master RNG seed (metric benchmark only).",
)
@click.option(
    "--ancestry-seed",
    type=int,
    default=1,
    show_default=True,
    help="Ancestry seed (perf benchmark only).",
)
@click.option(
    "--mutation-seed",
    type=int,
    default=2,
    show_default=True,
    help="Mutation seed (perf benchmark only).",
)
def main(
    bench_type,
    num_samples,
    sequence_length,
    version,
    csv_path,
    num_replicates,
    random_seed,
    ancestry_seed,
    mutation_seed,
):
    if version == "0.5.0":
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
    elif version == "0.4.1":
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer-old")
    else:
        raise ValueError("Version must be either '0.5.0' or '0.4.1'")
    sys.path.insert(0, tsinfer_path)
    global tsinfer
    import tsinfer
        
    if bench_type.lower() == "metric":
        run_metric_benchmark(
            num_samples_list=list(num_samples),
            num_replicates=num_replicates,
            sequence_length=sequence_length,
            version=version,
            random_seed=random_seed,
            csv_path=csv_path,
        )
    else:
        run_perf_benchmark(
            num_samples_list=list(num_samples),
            sequence_length=sequence_length,
            version=version,
            csv_path=csv_path,
            ancestry_seed=ancestry_seed,
            mutation_seed=mutation_seed,
        )


if __name__ == "__main__":
    main()
