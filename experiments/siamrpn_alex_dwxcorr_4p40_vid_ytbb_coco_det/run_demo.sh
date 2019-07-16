export PYTHONPATH=/dockerdata/xmmtyding/GitHub/pysot:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ../../tools/demo_video.py \
    --config config.yaml \
    --snapshot snapshot/checkpoint_e29.pth \
    --video /dockerdata/xmmtyding/Card2.mp4 \
    --init_bbox 331 288 126 82
