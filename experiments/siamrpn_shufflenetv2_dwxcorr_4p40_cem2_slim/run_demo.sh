export PYTHONPATH=/Users/dingyang/Public/GitHub/pysot:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=3
python ../../tools/demo.py \
    --config config.yaml \
    --snapshot snapshot/checkpoint_e27.pth \
    --video /Users/dingyang/Downloads/xigua.mp4 \
