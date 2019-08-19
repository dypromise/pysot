#!/usr/bin/bash
export PYTHONPATH=/dockerdata/xmmtyding/GitHub/pysot:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7

DATASET=VOT2016
START=15
END=50
seq $START 1 $END | xargs -I {} echo "snapshot/checkpoint_e{}.pth" | xargs -I {} \
python -u ../../tools/test.py \
	--snapshot {} \
	--config config.yaml \
	--dataset ${DATASET} 2>&1 | tee logs/test_dataset.log
