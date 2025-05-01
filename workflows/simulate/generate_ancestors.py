import os
import sys
import click
import tskit
from pathlib import Path

@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--version", required=True, type=str)
@click.option("--threads", required=True, type=int)
@click.option("--data-dir", required=True, type=click.Path())


def generate_ancestors(input, output, version, threads, data_dir):
    """Generate ancestors from a tree sequence."""

    if version == "v1.0":
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
    elif version == "v0.4":
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer-0.4")
    else:
        raise ValueError("Version must be either 'v1.0' or 'v0.4'")

    sys.path.append(tsinfer_path)
    import tsinfer

    input = Path(input)
    output = Path(output)
    data_dir = Path(data_dir)

    ts = tskit.load(str(input))
    sample_data = tsinfer.SampleData.from_tree_sequence(ts)

    progress_dir = data_dir / "progress" / "generate_ancestors"
    progress_dir.mkdir(parents=True, exist_ok=True)
    log_file = progress_dir / f"{output.stem}.log"

    with open(log_file, "w") as log_f:
        ancestors = tsinfer.generate_ancestors(
            sample_data,
            path=str(output),
            genotype_encoding=1,
            num_threads=threads,
            progress_monitor=tsinfer.progress.ProgressMonitor(
                tqdm_kwargs={"file": log_f, "mininterval": 30}
            ),
        )
        if ancestors.num_ancestors == 0:
            raise ValueError("No ancestors generated")
        if ancestors.num_sites == 0:
            raise ValueError("No sites generated")

if __name__ == "__main__":
    generate_ancestors()


