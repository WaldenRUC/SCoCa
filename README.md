# SCoCa Version 3.0

需要补充以下数据文件：（tiangong暂时不需补充）
1. ./BERT/BERTChinese/
2. ./BERT/BERTModel/
3. ./CLModel/BertContrastive.aol
4. ./CLModel/BertContrastive.tiangong
5. ./SCL/data/aol/train.pos.txt
6. ./SCL/data/aol/dev.pos.txt
7. ./SCL/data/tiangong/train.pos.txt
8. ./SCL/data/tiangong/dev.pos.txt
9. ./Ranking/data/aol/train_line.txt
10. ./Ranking/data/aol/test_line.middle.txt
11. ./Ranking/data/aol/test_line.txt
12. ./Ranking/data/tiangong/train.point.txt
13. ./Ranking/data/tiangong/dev.point.txt
14. ./Ranking/data/tiangong/test.point.lastq.txt
15. ./Ranking/data/tiangong/test.point.preq.txt

运行SCL和Ranking脚本：需要修改source命令后的环境参数

Ranking脚本中，pretrain_model_path需要改成SCL预训练相应的路径文件名

requirements:

- Python 3.8.0
- Pytorch 1.10.2 (with GPU support)
- pytrec-eval 0.5
- Transformers 4.2.0
- setproctitle 1.3.1

运行：
```
bash Ranking.sh
```
或
```
bash SCL.sh
```
