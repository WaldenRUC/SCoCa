source /data1/zhaoheng_huang/anaconda3/bin/activate /data1/zhaoheng_huang/anaconda3/envs/coca
python -u ./SCL/runBertContras.py \
    --training \
    --task aol \
    --device_id 0 \
    --per_gpu_batch_size 128 \
    --per_gpu_test_batch_size 256 \
    --multiGPU "All" \
    --tqdm \
    --epochs 5 \
    --data_dir /data1/zhaoheng_huang/COCA/SCL/ \
    --hint "Point_task12_128_8"
python -u ./Notify/notify.py \
    --text "164训练完毕【Point_128_6】"