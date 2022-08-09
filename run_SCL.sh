source /home/douzc/anaconda3/bin/activate /home/douzc/anaconda3/envs/py38
python -u ./SCL/runBertContras.py \
    --task aol \
    --bert_model_path ./BERT/BERTModel \
    --device_id 0 \
    --pretrain_model_path ./CLModel/BertContrastive.aol \
    --per_gpu_batch_size 80 \
    --per_gpu_test_batch_size 80 \
    --learning_rate 4e-5 \
    --scheduler_used \
    --tqdm \
    --training \
    --multiGPU \
    --hint "qRep_coClick_randMask"
# 如果用全部的gpu训练，则把is_multiGPU行首注释去掉即可，且把device_id改为0