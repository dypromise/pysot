export PYTHONPATH=/dockerdata/xmmtyding/GitHub/pysot:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ../../tools/demo_video.py \
    --config config.yaml \
    --snapshot snapshot/checkpoint_e39.pth \
    --video /dockerdata/xmmtyding/ft_local/Coke.avi \
    --init_bbox 298 159 46 82
