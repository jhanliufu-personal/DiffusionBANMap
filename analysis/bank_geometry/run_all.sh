#!/bin/bash
# Full bank-geometry pipeline (resumable: finished steps are skipped). Run from anywhere.
#   DBM_CKPT=latest ./analysis/bank_geometry/run_all.sh      # also runs the diffusion cycle test
#   PYTHON=python ./analysis/bank_geometry/run_all.sh        # use the active (e.g. conda) env instead of uv;
#                                                            # then `pip install "rembg[cpu]"` once for the cut-outs
set -e
cd "$(dirname "$0")/../.."
L=outputs/bank_geometry/logs
mkdir -p $L
PYTHON=${PYTHON:-"uv run python"}
PY="$PYTHON -m analysis.bank_geometry"
if [ "$PYTHON" = "uv run python" ]; then REMBG_PY='uv run --with rembg[cpu] python'; else REMBG_PY=$PYTHON; fi

# 0. decoder-bench out-of-fold predictions that q1/q2/dbm_cycle read (skipped if already present)
for t in meanpatch cls; do
  need=""
  for m in ridge rescale_per_dim snap_bank bayes_gmm32_bank bayes_bank_posterior bayes_bank_posterior_map; do
    [ -f outputs/decoder_bench/preds_${t}_${m}.npy ] || need="$need $m"
  done
  if [ -n "$need" ]; then $PYTHON -m analysis.decoder_bench.exp1_methods $t $need 2>&1 | tee -a $L/decoder_bench_$t.log; fi
done

# 1. embeddings (GPU). Cut-outs need rembg; they are skipped if already on disk.
$PY.build_imagesets edits            2>&1 | tee $L/edits.log
$PY.build_imagesets inet_list        2>&1 | tee $L/inet_list.log
$REMBG_PY -m analysis.bank_geometry.build_imagesets inet_cutout 2>&1 | tee $L/cutout.log
$PY.build_imagesets inet_raw         2>&1 | tee $L/inet_raw.log
$PY.build_imagesets inet_styled      2>&1 | tee $L/inet_styled.log
$PY.build_imagesets inet_gray        2>&1 | tee $L/inet_gray.log

# 2. diffusion cycle consistency + DBM-sampled bank (needs the finetuned meanpatch checkpoint)
if [ -n "$DBM_CKPT" ]; then
  $PY.dbm_cycle --config config/diffusion_full_stimuli_finetune_flow.yaml --ckpt "$DBM_CKPT" \
      --n 200 --n_seeds 2 --steps 50 --bank_sample 5000 2>&1 | tee $L/dbm_cycle.log
fi

# 3. analyses (CPU, numpy). Independent, so run in parallel.
$PY.q1_convexity meanpatch cls > $L/q1.log 2>&1 &
$PY.q2_composition            > $L/q2.log 2>&1 &
$PY.q3_bank_choice            > $L/q3.log 2>&1 &
wait
echo "ALL DONE: outputs/bank_geometry/{q1,q2,q3}_results.json, figs/"
