#!/usr/bin/bash
export PYTHONPATH=/dockerdata/xmmtyding/GitHub/pysot:$PYTHONPATH

DATASET=VOT2016
START=10
END=100
seq $START 1 $END | xargs -I {} echo "snapshot/checkpoint_e{}.pth" | xargs -I {} \
python -u ../../tools/test.py \
	--snapshot {} \
	--config config.yaml \
	--dataset ${DATASET} 2>&1 | tee logs/test_dataset.log
