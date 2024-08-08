#!/usr/bin/env bash

module load anaconda
conda activate solarski
cd /work/asolarski/GECEval
#srun python3 run_llm.py --model tower7B --iterations 5
srun python3 run_llm.py --model gemma2B --iterations 1

