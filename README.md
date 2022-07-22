# SCoCa

需要补充以下数据文件：
1. ./BERT/BERTChinese/
2. ./BERT/BERTModel/
3. ./CLModel/BertContrastive.aol
4. ./CLModel/BertContrastive.tiangong
5. ./data/aol/train.pos.txt
6. ./data/aol/dev.pos.txt
7. ./data/tiangong/train.pos.txt
8. ./data/tiangong/dev.pos.txt


运行脚本：需要修改source命令后的参数

requirements:

- Python 3.8.0
- Pytorch 1.10.2 (with GPU support)
- pytrec-eval 0.5
- Transformers 4.2.0