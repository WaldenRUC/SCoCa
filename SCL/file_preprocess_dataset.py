import torch
import linecache
from torch.utils.data import Dataset
import numpy as np
import random
import re
contras_sep_token = "[####]"
class ContrasDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, aug_strategy = ["sent_deletion", "term_deletion", "qd_reorder"]):
        super(ContrasDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._rnd = random.Random(0)
        self._aug_strategy = aug_strategy
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def check_length(self, pairlist):   #(int)[q1, d1, q2, d2, ...]
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:          #(int)[q1, d1]
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:                           #(int)[q1, d1, q2, d2, ...]
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist

    def anno_main(self, qd_pairs):
        #*qd_pairs: (str)[q1, d1, q2, d2, ...] (i.e.) (str)[[w1, w2, ..], [w1, w2, ..], [...], [...]]
        all_qd = []
        for qd in qd_pairs: #*      qd: [w1, w2, ...]
            qd = self._tokenizer.tokenize(qd)
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)
        history = all_qd[:-2]
        #print(history)
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        history_toks = ["[CLS]"]
        segment_ids = [0]
        for iidx, sent in enumerate(history):
            #print(sent)
            history_toks.extend(sent + ["[eos]"])
            segment_ids.extend([0] * (len(sent) + 1))
        query_tok += ["[eos]"]
        query_tok += ["[SEP]"]
        doc_tok += ["[eos]"]
        doc_tok += ["[SEP]"]
        all_qd_toks = history_toks + query_tok + doc_tok
        segment_ids.extend([0] * len(query_tok))
        segment_ids.extend([0] * len(doc_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        #print(all_qd_toks)
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids

    def check_length_endswith_q(self, pairlist):
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 1:
            while len(pairlist[0]) + 2 > max_seq_length:
                pairlist[0].pop(0)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length_endswith_q(pairlist)
        return pairlist

    def anno_endswith_q(self, qd_pairs):
        #*qd_pairs: (str)[q1, d1, q2, d2, ..., qn] (i.e.) (str)[[w1, w2, ..], [w1, w2, ..], [...], [...]]
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)
            all_qd.append(qd)
        all_qd = self.check_length_endswith_q(all_qd)
        history = all_qd[:-1]
        query_tok = all_qd[-1]
        #* history_toks和segment_ids初始设定
        history_toks = ["[CLS]"]
        segment_ids = [0]
        for _, sent in enumerate(history):
            history_toks.extend(sent + ["[eos]"])
            segment_ids.extend([0] * (len(sent) + 1))
        query_tok += ["[eos]"]
        query_tok += ["[SEP]"]
        all_qd_toks = history_toks + query_tok
        segment_ids.extend([0] * len(query_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids

    def _term_deletion(self, sent, ratio=0.6):
        tokens = sent.split()
        num_to_delete = int(round(len(tokens) * ratio))
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        self._rnd.shuffle(cand_indexes)
        output_tokens = list(tokens)
        deleted_terms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(deleted_terms) >= num_to_delete:
                break
            if len(deleted_terms) + len(index_set) > num_to_delete:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_token = "[term_del]"
                output_tokens[index] = masked_token
                deleted_terms.append((index, tokens[index]))
        assert len(deleted_terms) <= num_to_delete
        return " ".join(output_tokens)
    def augmentation(self, sequence, strategy):
        sent_del_ratio = 0.6
        random_positions = -1
        if strategy == "sent_deletion":     #! 随机删除q或d, 删除后用[sent_del]替代
            random_num = int(len(sequence) * sent_del_ratio)
            random_positions = self._rnd.sample(list(range(len(sequence))), random_num)
            for random_position in random_positions:
                sequence[random_position] = "[sent_del]"
            aug_sequence = sequence
        elif strategy == "term_deletion":
            aug_sequence = []
            for sent in sequence:
                sent_aug = self._term_deletion(sent)
                sent_aug += " "
                sent_aug = re.sub(r'(\[term_del\] ){2,}', "[term_del] ", sent_aug)
                sent_aug = sent_aug[:-1]
                aug_sequence.append(sent_aug)
        elif strategy == "qd_reorder":          #! [q1, d1, q2, d2, ...] => [q2, d2, q1, d1, ...]
            change_pos = self._rnd.sample(list(range(len(sequence) // 2)), 2)
            aug_sequence = sequence.copy()
            tmp = sequence[change_pos[1] * 2:change_pos[1] * 2 + 2]
            aug_sequence[change_pos[1] * 2:change_pos[1] * 2 + 2] = sequence[change_pos[0] * 2:change_pos[0] * 2 + 2]
            aug_sequence[change_pos[0] * 2:change_pos[0] * 2 + 2] = tmp
        else:
            assert False
        return aug_sequence, random_positions
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)       # 获取指定行号的数据；linecache将对文件的操作映射到内存中
        line = line.strip().split(contras_sep_token)
        assert len(line) == 3       # version: 0815
        qd_pairs1 = line[1].strip()
        qd_pairs2 = line[2].strip()
        qd_pairs1 = qd_pairs1.split("\t")   #[q1, d1, q2, d2, ....]
        qd_pairs2 = qd_pairs2.split("\t")   #[q1, d1, q2, d2, ....]
        if line[0] == "Point":      # 自我掩码对比学习，同coca
            random_qd_pairs1 = qd_pairs1.copy()
            random_qd_pairs2 = qd_pairs2.copy()
            if len(qd_pairs1) <= 2:
                aug_strategy = ["sent_deletion", "term_deletion"]
            else:
                aug_strategy = self._aug_strategy
            strategy1 = self._rnd.choice(aug_strategy)
            random_qd_pairs1, random_pos1 = self.augmentation(random_qd_pairs1, strategy1)
            strategy2 = self._rnd.choice(aug_strategy)
            random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy2)
            while random_pos1 == random_pos2 or strategy1 == strategy2:
                strategy2 = self._rnd.choice(aug_strategy)
                random_qd_pairs2 = qd_pairs2.copy()
                random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy2)
            input_ids, attention_mask, segment_ids = self.anno_main(random_qd_pairs1)
            input_ids2, attention_mask2, segment_ids2 = self.anno_main(random_qd_pairs2)
            batch = {
                'input_ids1': input_ids,
                'token_type_ids1': segment_ids,
                'attention_mask1': attention_mask,
                'input_ids2': input_ids2,
                'token_type_ids2': segment_ids2,
                'attention_mask2': attention_mask2,
            }
            return batch
        elif line[0] == "qRep":
            # after sampling, apply a term deletion to enhance robustness
            aug_sequence = []
            for index, sent in enumerate(qd_pairs2):
                if index % 2 == 0:  # q
                    aug_sequence.append(sent)
                else:       # d
                    sent_aug = self._term_deletion(sent)
                    sent_aug += " "
                    sent_aug = re.sub(r'(\[term_del\] ){2,}', "[term_del] ", sent_aug)
                    sent_aug = sent_aug[:-1]
                    aug_sequence.append(sent_aug)
            input_ids, attention_mask, segment_ids = self.anno_main(qd_pairs1)
            input_ids2, attention_mask2, segment_ids2 = self.anno_main(aug_sequence)
            batch = {
                'input_ids1': input_ids,
                'token_type_ids1': segment_ids,
                'attention_mask1': attention_mask,
                'input_ids2': input_ids2,
                'token_type_ids2': segment_ids2,
                'attention_mask2': attention_mask2,
            }
            return batch
        elif line[0] == "coClick":
            assert len(qd_pairs1) % 2 == 1 and len(qd_pairs2) % 2 == 1
            input_ids, attention_mask, segment_ids = self.anno_endswith_q(qd_pairs1)
            input_ids2, attention_mask2, segment_ids2 = self.anno_endswith_q(qd_pairs2)
            batch = {
                'input_ids1': input_ids,
                'token_type_ids1': segment_ids,
                'attention_mask1': attention_mask,
                'input_ids2': input_ids2,
                'token_type_ids2': segment_ids2,
                'attention_mask2': attention_mask2,
            }
            return batch
        else:
            assert False

    def __len__(self):
        return self._total_data