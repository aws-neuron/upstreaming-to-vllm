#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -x

export MODEL_PATH="$HOME/models/llama-3.2-3b"
export COMPILED_MODEL_PATH="$HOME/compiled_models/llama32di_compiled"

inference_demo \
    --model-type llama \
    --task-type causal-lm \
    run \
    --model-path $MODEL_PATH \
    --compiled-model-path $COMPILED_MODEL_PATH \
    --torch-dtype bfloat16 \
    --tp-degree 16 \
    --batch-size 2 \
    --ctx-batch-size 1 \
    --tkg-batch-size 2 \
    --is-continuous-batching \
    --max-context-length 256 \
    --seq-len 256 \
    --on-device-sampling \
    --prompt "What is annapurna labs?" \
    --save-sharded-checkpoint \
    --apply-seq-ids-mask \
    --compile-only
