import torch
import linecache
from torch.utils.data import Dataset
import numpy as np
import random
import re
contras_sep_token = "[####]"
dup_rate = 0.3
class ContrasDataset(Dataset):
    def __init__(
        self, filename, max_seq_length, tokenizer, q2coq_deep, q2coq_wide, d2cod_deep, d2cod_wide, CandSession,
        deletion_ratio = 0.6, replace_ratio = 0.5,
        aug_strategy = ["sent_deletion", "term_deletion", "qd_reorder", "qd_replace_deep", "qd_replace_wide", "qd_dup"]
    ):
        super(ContrasDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._rnd = random.Random(0)
        self._deletion_ratio = deletion_ratio
        self._replace_ratio = replace_ratio
        self._aug_strategy = aug_strategy
        self._q2coq_deep = q2coq_deep
        self._q2coq_wide = q2coq_wide
        self._d2cod_deep = d2cod_deep
        self._d2cod_wide = d2cod_wide
        self._CandSession = CandSession
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    def check_length(self, pairlist):       # pairlist: (int)[q1, d1, q2, d2, ...]
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:              # pairlist: (int)[q1, d1]
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:                               # pairlist: (int)[q1, d1, q2, d2, ...]
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist
    def anno_main(self, qd_pairs):
        # qd_pairs: (str)[q1, d1, q2, d2, ...] 
        # (i.e.) (str)[ [w1, w2, ..], [w1, w2, ..], [...], [...] ]
        all_qd = []
        for qd in qd_pairs:                 # qd: [w1, w2, ...]
            qd = self._tokenizer.tokenize(qd)
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)  # [q1, d1, ..., qc, dc]    
        history = all_qd[:-2]               # [q1, d1, ...]
        query_tok = all_qd[-2]              # qc
        doc_tok = all_qd[-1]                # dc
        history_toks = ["[CLS]"]
        segment_ids = [0]
        for iidx, sent in enumerate(history):
            history_toks.extend(sent + ["[eos]"])       # [ [cls], q1, [eos] ,d1, [eos], ...]
            segment_ids.extend([0] * (len(sent) + 1))   
        query_tok += ["[eos]"]              # [qc, [eos]]
        doc_tok += ["[eos]"]
        doc_tok += ["[SEP]"]                # [dc, [eos], [sep]]
        all_qd_toks = history_toks + query_tok + doc_tok        # [ [cls], q1, [eos], d1, [eos], ..., qc, [eos], dc, [eos], [sep] ]
        segment_ids.extend([0] * len(query_tok))
        segment_ids.extend([0] * len(doc_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:          # padding
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids
    def _term_deletion(self, sent):       # sent: q/d: "w1 w2 ..."
        ratio=self._deletion_ratio
        tokens = sent.split()      # tokens: [w1, w2, ...]
        num_to_delete = int(round(len(tokens) * ratio))
        cand_indexes = []
        for (i, token) in enumerate(tokens):            # term_del将其后面跟随的词尾(如##ing)也遮盖住
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        # cand_indexes: [[0], [1, 2], [3], [4, 5], ...]
        self._rnd.shuffle(cand_indexes)
        output_tokens = list(tokens)        # output_tokens为tokens的一个副本，操作副本不会改变tokens的数据
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
        sent_del_ratio = self._deletion_ratio
        random_positions = -1
        if strategy == "sent_deletion":     # 随机删除q或d, 删除后用[sent_del]替代
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
        elif strategy == "qd_reorder":          # [q1, d1, q2, d2, ...] => [q2, d2, q1, d1, ...]
            change_pos = self._rnd.sample(list(range(len(sequence) // 2)), 2)
            aug_sequence = sequence.copy()
            tmp = sequence[change_pos[1] * 2:change_pos[1] * 2 + 2]
            aug_sequence[change_pos[1] * 2:change_pos[1] * 2 + 2] = sequence[change_pos[0] * 2:change_pos[0] * 2 + 2]
            aug_sequence[change_pos[0] * 2:change_pos[0] * 2 + 2] = tmp
        elif strategy == "qd_replace_deep":          # [q1, d1, q2, d2, ...] => [q'1, d'1, q'2, d'2, ...]
            aug_sequence = []
            is_query_only = (random.random() <= 0.5)
            for index, sent in enumerate(sequence):
                if is_query_only == True and index % 2 == 0:  # query
                    co_sent = self._q2coq_deep.get(sent, sent)
                    if type(co_sent) == list:
                        if random.random() <= self._replace_ratio:
                            aug_sequence.append(random.choice(co_sent))
                        else:
                            aug_sequence.append(sent)
                    elif type(co_sent) == str:
                        aug_sequence.append(sent)
                    else: assert False
                elif is_query_only == False and index % 2 == 1:   # doc
                    co_sent = self._d2cod_deep.get(sent, sent)
                    if type(co_sent) == list:
                        if random.random() <= self._replace_ratio:
                            aug_sequence.append(random.choice(co_sent))
                        else:
                            aug_sequence.append(sent)
                    elif type(co_sent) == str:
                        aug_sequence.append(sent)
                    else: assert False
                else:
                    aug_sequence.append(sent)
        elif strategy == "qd_replace_wide":     # [q1, d1, q2, d2, ...] => [q'1, d'1, q'2, d'2, ...]
            aug_sequence = []
            is_query_only = (random.random() <= 0.5)
            for index, sent in enumerate(sequence):
                if is_query_only == True and index % 2 == 0:  # query
                    sim_sent = self._q2coq_wide.get(sent, sent)
                    if type(sim_sent) == list:
                        if random.random() <= self._replace_ratio:
                            aug_sequence.append(random.choice(sim_sent))
                        else:       # 不变
                            aug_sequence.append(sent)
                    elif type(sim_sent) == str:
                        aug_sequence.append(sent)
                    else: assert False
                elif is_query_only == False and index % 2 == 1:    # doc
                    sim_sent = self._d2cod_wide.get(sent, sent)
                    if type(sim_sent) == list:
                        if random.random() <= self._replace_ratio:
                            aug_sequence.append(random.choice(sim_sent))
                        else:
                            aug_sequence.append(sent)
                    elif type(sim_sent) == str:
                        aug_sequence.append(sent)
                    else: assert False
                else:
                    aug_sequence.append(sent)
        elif strategy == "qd_dup":
            aug_sequence = []
            q_index = [2*i for i in range(int(len(sequence)/2))]     # [0, 2, 4, 6, 8, ...]
            dup_q_index = random.sample(q_index, max(1, int(len(sequence)/2*dup_rate))) #[0, 6, ...]
            for qID in q_index:
                aug_sequence.append(sequence[qID])
                aug_sequence.append(sequence[qID+1])
                if qID in dup_q_index:      # 复制
                    aug_sequence.append(sequence[qID])
                    aug_sequence.append(sequence[qID+1])
        else:
            assert False
        return aug_sequence, random_positions
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)       # 获取指定行号的数据；linecache将对文件的操作映射到内存中
        line = line.strip().split(contras_sep_token)
        assert len(line) == 3
        qd_pairs1 = line[1].strip()
        qd_pairs2 = line[2].strip()
        qd_pairs1 = qd_pairs1.split("\t")   #[q1, d1, q2, d2, ....]
        qd_pairs2 = qd_pairs2.split("\t")   #[q1, d1, q2, d2, ....]
        assert len(qd_pairs1) % 2 == 0 and len(qd_pairs2) % 2 == 0
        if line[0] == "COCA":
            random_qd_pairs1 = qd_pairs1.copy()
            random_qd_pairs2 = qd_pairs2.copy()
            if len(qd_pairs1) <= 2:
                aug_strategy = ["sent_deletion", "term_deletion", "qd_replace_deep", "qd_replace_wide"]#, "qd_overlap", "qd_dup"]
            else:
                #aug_strategy = self._aug_strategy
                aug_strategy = ["sent_deletion", "term_deletion", "qd_reorder", "qd_replace_deep", "qd_replace_wide"]#, "qd_overlap", "qd_dup"]
            
            strategy1 = self._rnd.choice(aug_strategy)
            random_qd_pairs1, random_pos1 = self.augmentation(random_qd_pairs1, strategy1)
            strategy2 = self._rnd.choice(aug_strategy)
            random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy2)
            #! random_pos1, random_pos2似乎有bug，如果都为-1，则会继续选择strategy，直到其中一个strategy为term del
            while ((random_pos1 != -1 or random_pos2 != -1) and random_pos1 == random_pos2) or strategy1 == strategy2 or random_qd_pairs1 == random_qd_pairs2:
                strategy1 = self._rnd.choice(aug_strategy)
                strategy2 = self._rnd.choice(aug_strategy)
                random_qd_pairs1 = qd_pairs1.copy()
                random_qd_pairs1, random_pos1 = self.augmentation(random_qd_pairs1, strategy1)
                random_qd_pairs2 = qd_pairs2.copy()
                random_qd_pairs2, random_pos2 = self.augmentation(random_qd_pairs2, strategy2)

            input_ids, attention_mask, segment_ids = self.anno_main(random_qd_pairs1)
            input_ids2, attention_mask2, segment_ids2 = self.anno_main(random_qd_pairs2)
            
            # 第三个句子为强负例 [可能没有强负例]
            # neg_qd_pairs_list = self._CandSession["\t".join(qd_pairs1)]
            # 0927: 随机从前缀中选一系列cand seq来替代
            prefix_session = list(range(2, len(qd_pairs1) + 2, 2))      # [2, 4, 6, ..., length]
            prefix_session_index = random.choice(prefix_session)
            neg_qd_pairs_list = self._CandSession["\t".join(qd_pairs1[:prefix_session_index])]
            if neg_qd_pairs_list == []:      # 这个batch就不算了（只有dev集会出现这样的问题）
                return None
            neg_qd_pairs = random.choice(neg_qd_pairs_list)     #[q, d, q, d]
            neg_qd_pairs = neg_qd_pairs + qd_pairs1[prefix_session_index:]
            input_ids_neg, attention_mask_neg, token_type_ids_neg = self.anno_main(neg_qd_pairs)
            batch = {
                'input_ids1': input_ids,
                'token_type_ids1': segment_ids,
                'attention_mask1': attention_mask,
                'input_ids2': input_ids2,
                'token_type_ids2': segment_ids2,
                'attention_mask2': attention_mask2,
                'input_ids_neg': input_ids_neg,
                'token_type_ids_neg': token_type_ids_neg,
                'attention_mask_neg': attention_mask_neg
            }
            return batch
        else:
            assert False

    def __len__(self):
        return self._total_data