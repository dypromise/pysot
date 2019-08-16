export PYTHONPATH=/Users/dingyang/Public/GitHub/pysot:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=3
python ../../tools/demo.py \
    --config config.yaml \
    --snapshot snapshot/checkpoint_e42.pth \
    --video /Users/dingyang/Downloads/card2.mp4 \
