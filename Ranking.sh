source /data1/zhaoheng_huang/anaconda3/bin/activate /data1/zhaoheng_huang/anaconda3/envs/coca
python -u ./Ranking/runBert.py \
    --task aol \
    --per_gpu_batch_size 32 \
    --per_gpu_test_batch_size 64 \
    --pretrain_model_path /data1/zhaoheng_huang/COCA/SCL/model/BertContrastive.aol.5.10.128.Point_task12_128_6 \
    --tqdm \
    --multiGPU "All" \
    --hint "Ranking" \
    --data_dir /data1/zhaoheng_huang/COCA/Ranking/