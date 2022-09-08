import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class BertContrastive(nn.Module):
    def __init__(self, bert_model, temperature, args):
        super(BertContrastive, self).__init__()
        self.bert_model = bert_model
        self.temperature = temperature
        self.cl_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.args = args
    def forward(self, batch_data):
        batch_size = batch_data["input_ids1"].size(0)

        input_ids1 = batch_data["input_ids1"]
        attention_mask1 = batch_data["attention_mask1"]
        token_type_ids1 = batch_data["token_type_ids1"]
        bert_inputs1 = {'input_ids': input_ids1, 'attention_mask': attention_mask1, 'token_type_ids': token_type_ids1}
        sent_rep1 = self.bert_model(**bert_inputs1)[1]

        input_ids2 = batch_data["input_ids2"]
        attention_mask2 = batch_data["attention_mask2"]
        token_type_ids2 = batch_data["token_type_ids2"]
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}
        sent_rep2 = self.bert_model(**bert_inputs2)[1]
        
        sent_norm1 = sent_rep1.norm(dim=-1, keepdim=True)  # sqrt(x1^2 + x2^2 + ...)
        sent_norm2 = sent_rep2.norm(dim=-1, keepdim=True)
        # sent_rep1: [bs, 768]; sent_norm1: [bs, 1]
        # sent_rep2: [bs, 768]; sent_norm2: [bs, 1]
        return sent_rep1, sent_norm1, sent_rep2, sent_norm2
        #! sent_rep1 & sent_rep2 对应第一维下标的两个向量为正例，
        #! 相乘得到的[bs, bs]矩阵中，对角上的是正例
        batch_self_11 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm1) + 1e-6)
        batch_cross_12 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm2) + 1e-6)  # [batch, batch]
        batch_self_11 = batch_self_11 / self.temperature
        batch_cross_12 = batch_cross_12 / self.temperature
        #! self_11 和 cross_12: 算出s_i和s_j的余弦相似度(s_i * s_j / (|s_i|*|s_j|))
        """batch:
        s11     s12
        s21     s22
        s31     s32
        ...
        sn1     sn2
        ---
        self_11算的是s11~sn1这一列的相似度矩阵
        cross_12算的是s11~sn1与s12~sn2的相似度矩阵
        self_22算的是s12~sn2这一列的相似度矩阵
        cross_21算的是s12~sn2与s11~sn1的相似度矩阵
        """
        batch_first = torch.cat([batch_self_11, batch_cross_12], dim=-1)  # [batch, batch * 2]
        if self.args.multiGPU == "False":    # 单卡
            batch_arange = torch.arange(batch_size).to("cuda:%s"%(self.args.device_id))
        else:
            batch_arange = torch.arange(batch_size).to(torch.cuda.current_device())
        mask = F.one_hot(batch_arange, num_classes=batch_size * 2) * -1e10
        """mask:
        [-1e10, 0, 0, 0, 0, 0, 0, 0],
        [0, -1e10, 0, 0, 0, 0, 0, 0],
        [0, 0, -1e10, 0, 0, 0, 0, 0],
        [0, 0, 0, -1e10, 0, 0, 0, 0]
        """
        batch_first += mask
        batch_label1 = batch_arange + batch_size  # [batch]
        """batch_label1:
        [128, 129, 130, ..., 255]
        """
        batch_self_22 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm2) + 1e-6)  # [batch, batch]
        batch_cross_21 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm1) + 1e-6)  # [batch, batch]
        batch_self_22 = batch_self_22 / self.temperature
        batch_cross_21 = batch_cross_21 / self.temperature
        batch_second = torch.cat([batch_self_22, batch_cross_21], dim=-1)  # [batch, batch * 2]
        batch_second += mask
        batch_label2 = batch_arange + batch_size  # [batch]
        """batch_label2:
        [128, 129, 130, ..., 255]
        """

        #TODO 在外面算loss

        batch_predict = torch.cat([batch_first, batch_second], dim=0)
        batch_label = torch.cat([batch_label1, batch_label2], dim=0)  # [batch * 2]

        contras_loss = self.cl_loss(batch_predict, batch_label)     #多分类
        batch_logit = batch_predict.argmax(dim=-1)
        acc = torch.sum(batch_logit == batch_label).float() / (batch_size * 2)

        return contras_loss, acc
