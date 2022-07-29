source /data00/zhaoheng_huang/anaconda3/bin/activate py38
python -u ./SCL/runBertContras.py \
    --task aol \
    --bert_model_path ./BERT/BERTModel \
    --device_id 0 \
    --pretrain_model_path ./CLModel/BertContrastive.aol \
    --per_gpu_batch_size 128 \
    --per_gpu_test_batch_size 128 \
    --learning_rate 5e-5 \
    --scheduler_used \
    --tqdm \
    --training \
    --multiGPU
# 如果用全部的gpu训练，则把is_multiGPU行首注释去掉即可，且把device_id改为0