source /home/douzc/anaconda3/bin/activate /home/douzc/anaconda3/envs/py38
python -u ./Ranking/runBert.py \
    --task aol \
    --bert_model_path ./BERT/BERTModel \
    --pretrain_model_path ./SCL/model/BertContrastive.aol.4.10.80.qRep_coClick_randMask \
    --device_id 0 \
    --per_gpu_batch_size 128 \
    --per_gpu_test_batch_size 128 \
    --scheduler_used \
    --tqdm \
    --training \
    --multiGPU \
    --hint "qRep_coClick_randMask"
#pretrain_model_path改成相应的路径文件名