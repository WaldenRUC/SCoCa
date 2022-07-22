source /data00/zhaoheng_huang/anaconda3/bin/activate py38
python -u runBert.py \
    --task aol \
    --bert_model_path ../BERT/BERTModel \
    --pretrain_model_path ../SCL/model/BertContrastive.aol \
    --device_id 0 \
    --is_training 1 \
    --is_multiGPU 1 \
    --per_gpu_batch_size 128 \
    --per_gpu_test_batch_size 128
#pretrain_model_path改成相应的路径文件名