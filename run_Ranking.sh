source /data00/zhaoheng_huang/anaconda3/bin/activate py38
python -u ./Ranking/runBert.py \
    --task aol \
    --bert_model_path ./BERT/BERTModel \
    --pretrain_model_path ./SCL/model/BertContrastive.aol.4.10.128 \
    --device_id 0 \
    --per_gpu_batch_size 64 \
    --per_gpu_test_batch_size 64 \
    --scheduler_used \
    --tqdm \
    --training \
    --multiGPU
#pretrain_model_path改成相应的路径文件名