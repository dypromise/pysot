#! /bin/bash
export PYTHONPATH=/dockerdata/xmmtyding/GitHub/pysot:$PYTHONPATH

set -e
DATASET=VOT2016
START=10
END=50
seq $START 1 $END | xargs -I {} python ../../tools/eval.py \
	--tracker_path ./results \
	--dataset ${DATASET} \
	--num 4 \
	--tracker_prefix 'checkpoint_e{}' | tee logs/eval_res.txt
