source /data00/zhaoheng_huang/anaconda3/bin/activate /data00/zhaoheng_huang/anaconda3/envs/py38
cd ContrastiveLearning
python -u runBertContras.py \
    --task aol \
    --per_gpu_batch_size 128 \
    --per_gpu_test_batch_size 256