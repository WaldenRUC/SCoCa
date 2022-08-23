source /data00/zhaoheng_huang/anaconda3/bin/activate /data00/zhaoheng_huang/anaconda3/envs/py38
python -u ./Ranking/runBert.py \
    --task aol \
    --per_gpu_batch_size 64 \
    --per_gpu_test_batch_size 128 \
    --pretrain_model_path ./SCL/model/BertContrastive.aol.4.10.128.COCA_pretrain_mixedSCL_collected_woSched \
    --tqdm \
    --multiGPU "0,1,2,3" \
    --hint "Ranking" \
    --data_dir /data00/zhaoheng_huang/COCA/Ranking/