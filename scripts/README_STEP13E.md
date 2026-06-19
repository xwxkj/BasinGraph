# Step 13E — CUTEst analysis and manuscript package

## Install

Copy the scripts into:

```text
~/Documents/BasinGraph202606/scripts/
```

## Run

```bash
cd ~/Documents/BasinGraph202606
conda activate basingraph-cutest
source protocols/cutest_env.sh

bash scripts/run_step13e_analysis.sh
```

## Expected completion message

```text
STEP_13E_OK
```

## Main output

```text
cutest_results/protocol_v2/analysis_final_v1/
cutest_results/protocol_v2/BasinGraph_CUTEst_FINAL_analysis_and_manuscript_inputs.zip
```

The analysis uses the frozen CUTEst protocol, 10,500 completed records and
all stored convergence histories.

### Important budget policy

CMA-ES and Multi-start L-BFGS-B may terminate early. This is allowed.
For expected-running-time calculations, an unsuccessful early termination
is charged the full prescribed budget; a successful early termination keeps
its observed time to target.
