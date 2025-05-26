#!/bin/bash
export PYTHONPATH=$(pwd)
snakemake --snakefile snakefiles/anc_eval.SnakeFile --directory . --cores all
