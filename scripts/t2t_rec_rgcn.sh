#!/bin/bash
let num_runs=$1
let gpu_id=$2

for i in $(seq 0 $((num_runs-1)));
do
    CUDA_VISIBLE_DEVICES=$gpu_id python -m parlai.tasks.redial.train_transformer_rec -mf saved/transformer_rec_both_rgcn_$i
done

