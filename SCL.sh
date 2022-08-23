source /data00/zhaoheng_huang/anaconda3/bin/activate /data00/zhaoheng_huang/anaconda3/envs/py38
python -u ./SCL/runBertContras.py \
    --training \
    --task aol \
    --device_id 0 \
    --multiGPU "All" \
    --hint "SCL" \
    --tqdm \
    --per_gpu_batch_size 128 \
    --per_gpu_test_batch_size 256 \
    --data_dir /data00/zhaoheng_huang/COCA/SCL/