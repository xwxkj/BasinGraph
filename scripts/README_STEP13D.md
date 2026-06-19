# Step 13D — Formal CUTEst benchmark

This package runs the seven-algorithm comparison on the frozen
`cutest_pre_registered_problem_list_v2.csv`.

## Protocol

- 50 pre-registered CUTEst instances
- 20 small, 20 medium, 10 large
- 21 fixed-dimensional and 29 scalable instances
- 30 paired seeds, executed as three resumable batches
- Algorithms:
  - BasinGraph
  - CMA-ES
  - true BIPOP-CMA-ES
  - Differential Evolution
  - Multi-start L-BFGS-B
  - Latin Hypercube Sampling
  - Random Search
- Evaluation budget:
  `min(20000, max(1000, 50*n))`

## Installation into the project

Copy all files into:

```text
~/Documents/BasinGraph202606/scripts/
```

## Run a first 10-seed batch

```bash
cd ~/Documents/BasinGraph202606
conda activate basingraph-cutest
source protocols/cutest_env.sh

bash scripts/run_step13d_cutest_batch.sh 0 10 4
```

## Run all 30 seeds

```bash
bash scripts/run_step13d_cutest_30seeds.sh 4
```

The runner is resumable. Existing complete problem–seed job files are skipped.

## Monitor progress

```bash
python scripts/step13d_check_status.py
```

The benchmark is quiet by default. Pass `--verbose` directly to the Python
runner only for debugging.

## Outputs

```text
cutest_results/protocol_v2/
    job_records/
    histories/
    job_failures/
    batch_metadata/
    progress.json
    cutest_raw_results_all_available.csv
```

Convergence histories are stored as compressed `.npz` files for later
performance-profile and data-profile analysis.
