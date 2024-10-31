# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这里的数据集针对的是entity type类别的数据
        所以数据集的sentence.txt一般就一句话，实体对已经用entity type给替代了...
   Author :        kedaxia
   date：          2021/12/22
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/22: 
-------------------------------------------------
"""

import re
import logging
from ipdb import set_trace
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from config import BertConfig
from src.utils.function_utils import get_pos_feature
from src.dataset_utils.data_process_utils import sequence_padding, InputExamples,make_graph_for_spacy,make_gpnn_graph_for_spacy,make_graph_for_neg
from nltk import sent_tokenize
import spacy

logger = logging.getLogger('main.entity_type_marker')



class NormalDataset(Dataset):
    def __init__(self, examples, config: BertConfig, tokenizer, label2id, device):
        super(NormalDataset, self).__init__()
        self.config = config
        self.examples = examples

        self.tokenizer = tokenizer
        # 将特殊实体加入到分词器，防止给切分

        self.label2id = label2id

        self.max_len = config.max_len
        self.device = device
        self.nlp =spacy.load("en_core_sci_lg")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        return self.examples[item]

    def collate_fn_predicate(self, features):
        '''
        这个专用于模型的predicate的collate_fn
        和collate_fn的不同是没有label的处理
        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_ent = []
        batch_max_len = 0
        for example in features:
            sent = example.text
            e1 = example.ent1_name
            e2 = example.ent2_name
            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)

            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

            e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            batch_ent.append({'e1':e1,'e2':e2})

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)

        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

        if 'gcn' in self.config.model_name or 'gat' in self.config.model_name:
            return  batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask,batch_ent

        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask

    def collate_fn(self, features):
        '''

        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_type = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_ent = []
        batch_max_len = 0
        for example in features:

            sent = example.text
            label = self.label2id[example.label]
            e1 = example.ent1_name
            e2 = example.ent2_name
            # e1_type = example.ent1_type
            # e2_type = example.ent2_type
            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)
            batch_labels.append(label)
            # set_trace()
            # if self.config.model_name == 'bioelectra_gpnn_entitymarker_model':
            #     subword_tokens = ' '.join(subword_tokens)
            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)
            # e1_type_res = self.tokenizer.encode(e1_type,add_special_tokens=False)
            # e2_type_res = self.tokenizer.encode(e2_type,add_special_tokens=False)
            # ent_type_res = self.tokenizer.encode(e1_type+' '+e2_type,add_special_tokens=False)
            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']
            if self.config.ent2_start_tag_id not in input_ids:
                print(example.text)
            e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)
            batch_ent.append({'e1':e1,'e2':e2})
            # batch_type.append(ent_type_res)
        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)
    
        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()
        # batch_type = torch.tensor(batch_type,device=self.device).long()
        batch_labels = torch.tensor(batch_labels, device=self.device).long() 

        if 'gcn' in self.config.model_name or 'gat' in self.config.model_name:
            return  batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels,batch_ent

        if self.config.scheme in [1,2]:
            return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels,batch_type
        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels

    def _process_seq_len(self, text, total_special_toks=3):
        """
            裁切句子的方法，直接使用clinicalTransformer提供的方法
            This function is used to truncate sequences with len > max_seq_len
            Truncate strategy:
            1. find all the index for special tags
            3. count distances between leading word to first tag and second tag to last.
            first -1- tag1 entity tag2 -2- last
            4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
            5. repeat until len is equal to max_seq_len
        """
        loop_count = 0
        while len(self.tokenizer.tokenize(text)) > (self.config.max_len - total_special_toks):
            text = self._truncate_helper(text)
            loop_count += 1
            if loop_count > 50:
                return
        return text

    def _truncate_helper(self, text):
        '''
        这是一个句子一个句子的找
        这里对原始的的text进行去除，并不是tokenize之后的....
        :param text:
        :return:
        '''
        tokens = text.split(" ")
        # 这是得到 word-level的index
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if
                                        tk.lower() in [self.config.ent1_start_tag, self.config.ent2_end_tag]]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0:  # 这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
            return text

        if truncate_space_head > truncate_space_tail:  # 如果离头更远，那么先抛弃头部的word...
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)
    
class NEGNormalDataset(Dataset):
    def __init__(self, examples, config: BertConfig, tokenizer, label2id, device):
        super(NEGNormalDataset, self).__init__()
        self.config = config
        self.examples = examples

        self.tokenizer = tokenizer
        # 将特殊实体加入到分词器，防止给切分

        self.label2id = label2id

        self.max_len = config.max_len
        self.neg_len = config.neg_len
        self.device = device
        self.nlp = self.nlp =spacy.load("en_core_sci_lg")
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        return self.examples[item]

    def collate_fn_predicate(self, features):
        '''
        这个专用于模型的predicate的collate_fn
        和collate_fn的不同是没有label的处理
        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_e1_neg_input_ids = []
        batch_e2_neg_input_ids = []
        batch_e1_neg_token_type_ids = []
        batch_e2_neg_token_type_ids = []
        batch_e1_neg_attention_masks = []
        batch_e2_neg_attention_masks = []
        batch_e1_neg_mask = []
        batch_e2_neg_mask = []
        batch_ent = []
        batch_max_len = 0
        batch_type = []
        for example in features:
            e1_neg = example.e1_neg
            e2_neg = example.e2_neg
            # e1_type = example.ent1_type
            # e2_type = example.ent2_type
            # 把所有的整合在一起

            all_e1_neg = e1_neg.split(' ')
            all_e2_neg = e2_neg.split(' ')
            for word in all_e2_neg:
                if word not in all_e1_neg:
                    all_e1_neg.append(word)
            e1_neg = ' '.join(all_e1_neg)
            sent = example.text
            # label = self.label2id[example.label]
            e1 = example.ent1_name
            e2 = example.ent2_name

            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)
            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)
            # ent_type_res = self.tokenizer.encode(e1_type+' '+e2_type,add_special_tokens=False)
            subword_tokens_1 = self.tokenizer.tokenize(e1_neg)
            subword_tokens_2 = self.tokenizer.tokenize(e2_neg)


            neg_encoder_res_1 = self.tokenizer.encode_plus(subword_tokens_1,truncation=True, max_length=self.neg_len,add_special_tokens=True)
            neg_encoder_res_2 = self.tokenizer.encode_plus(subword_tokens_2,truncation=True, max_length=self.neg_len,add_special_tokens=True)

            neg_input_ids_1 = neg_encoder_res_1['input_ids']
            neg_token_type_ids_1 = neg_encoder_res_1['token_type_ids']
            neg_attention_mask_1 = neg_encoder_res_1['attention_mask']

            neg_input_ids_2 = neg_encoder_res_2['input_ids']
            neg_token_type_ids_2 = neg_encoder_res_2['token_type_ids']
            neg_attention_mask_2 = neg_encoder_res_2['attention_mask']

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

            e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            e1_neg_mask = np.zeros(len(neg_input_ids_1))
            e2_neg_mask = np.zeros(len(neg_input_ids_2))
            e1_neg_mask[1:-1]=1
            e2_neg_mask[1:-1]=1
            


            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)

            batch_e1_neg_input_ids.append(neg_input_ids_1)
            batch_e1_neg_token_type_ids.append(neg_token_type_ids_1)
            batch_e1_neg_attention_masks.append(neg_attention_mask_1)
            batch_e2_neg_input_ids.append(neg_input_ids_2)
            batch_e2_neg_token_type_ids.append(neg_token_type_ids_2)
            batch_e2_neg_attention_masks.append(neg_attention_mask_2)
            batch_e1_neg_mask.append(e1_neg_mask)
            batch_e2_neg_mask.append(e2_neg_mask)
            batch_ent.append({'e1':e1,'e2':e2})
            # batch_type.append(ent_type_res)
        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()


        batch_e1_neg_input_ids = torch.tensor(sequence_padding(batch_e1_neg_input_ids, length=self.neg_len), device=self.device).long()
        batch_e1_neg_token_type_ids = torch.tensor(sequence_padding(batch_e1_neg_token_type_ids, length=self.neg_len), device=self.device).long()
        batch_e1_neg_attention_masks = torch.tensor(sequence_padding(batch_e1_neg_attention_masks, length=self.neg_len), device=self.device).long()
        batch_e2_neg_input_ids = torch.tensor(sequence_padding(batch_e2_neg_input_ids, length=self.neg_len), device=self.device).long()
        batch_e2_neg_token_type_ids = torch.tensor(sequence_padding(batch_e2_neg_token_type_ids, length=self.neg_len), device=self.device).long()
        batch_e2_neg_attention_masks = torch.tensor(sequence_padding(batch_e2_neg_attention_masks, length=self.neg_len), device=self.device).long()
        batch_e1_neg_mask = torch.tensor(sequence_padding(batch_e1_neg_mask, length=self.neg_len), device=self.device).long()
        batch_e2_neg_mask = torch.tensor(sequence_padding(batch_e2_neg_mask, length=self.neg_len), device=self.device).long()
        # batch_type = torch.tensor(batch_type,device=self.device).long()
        if 'gcn' in self.config.model_name or 'gat' in self.config.model_name:
            return  batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask,batch_e1_neg_input_ids,batch_e1_neg_token_type_ids,batch_e1_neg_attention_masks,batch_e2_neg_input_ids,batch_e2_neg_token_type_ids,batch_e2_neg_attention_masks,batch_e1_neg_mask,batch_e2_neg_mask,batch_ent
            
        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask,batch_e1_neg_input_ids,batch_e1_neg_token_type_ids,batch_e1_neg_attention_masks,batch_e2_neg_input_ids,batch_e2_neg_token_type_ids,batch_e2_neg_attention_masks,batch_e1_neg_mask,batch_e2_neg_mask

    def collate_fn(self, features):
        '''

        :param features:
        :return:
        '''
        raw_text_li = []
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_e1_mask = []
        batch_e2_mask = []
        batch_e1_neg_input_ids = []
        batch_e2_neg_input_ids = []
        batch_e1_neg_token_type_ids = []
        batch_e2_neg_token_type_ids = []
        batch_e1_neg_attention_masks = []
        batch_e2_neg_attention_masks = []
        batch_e1_neg_mask = []
        batch_e2_neg_mask = []
        batch_ent = []
        batch_max_len = 0
        batch_type = []
        for example in features:
            e1_neg = example.e1_neg
            e2_neg = example.e2_neg
            # e1_type = example.ent1_type
            # e2_type = example.ent2_type
            # 把所有的整合在一起

            all_e1_neg = e1_neg.split(' ')
            all_e2_neg = e2_neg.split(' ')
            for word in all_e2_neg:
                if word not in all_e1_neg:
                    all_e1_neg.append(word)
            e1_neg = ' '.join(all_e1_neg)
            sent = example.text
            label = self.label2id[example.label]
            e1 = example.ent1_name
            e2 = example.ent2_name

            # sent 是word-level list: ['Feadeal','ABVDF','the',...]
            raw_text_li.append(sent)
            subword_tokens = self.tokenizer.tokenize(sent)
            # 如果长度过长，那么开始裁剪长度
            if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

                logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
                sent = self._process_seq_len(sent)
                if not sent:
                    logger.warning('此数据难以裁剪，进行抛弃......')
                    continue
                subword_tokens = self.tokenizer.tokenize(sent)
                logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
                if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
                    continue

            if batch_max_len < len(subword_tokens):
                batch_max_len = len(subword_tokens)
            batch_labels.append(label)
            encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
                                                     add_special_tokens=True)

            subword_tokens_1 = self.tokenizer.tokenize(e1_neg)
            subword_tokens_2 = self.tokenizer.tokenize(e2_neg)


            neg_encoder_res_1 = self.tokenizer.encode_plus(subword_tokens_1,truncation=True, max_length=self.neg_len,add_special_tokens=True)
            neg_encoder_res_2 = self.tokenizer.encode_plus(subword_tokens_2,truncation=True, max_length=self.neg_len,add_special_tokens=True)

            neg_input_ids_1 = neg_encoder_res_1['input_ids']
            neg_token_type_ids_1 = neg_encoder_res_1['token_type_ids']
            neg_attention_mask_1 = neg_encoder_res_1['attention_mask']

            neg_input_ids_2 = neg_encoder_res_2['input_ids']
            neg_token_type_ids_2 = neg_encoder_res_2['token_type_ids']
            neg_attention_mask_2 = neg_encoder_res_2['attention_mask']

            input_ids = encoder_res['input_ids']
            token_type_ids = encoder_res['token_type_ids']
            attention_mask = encoder_res['attention_mask']

            e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
            e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
            e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
            e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

            e1_mask = np.zeros(len(input_ids))
            e2_mask = np.zeros(len(input_ids))
            e1_mask[e1_start_idx:e1_end_idx + 1] = 1
            e2_mask[e2_start_idx:e2_end_idx + 1] = 1

            e1_neg_mask = np.zeros(len(neg_input_ids_1))
            e2_neg_mask = np.zeros(len(neg_input_ids_2))
            e1_neg_mask[1:-1]=1
            e2_neg_mask[1:-1]=1
            


            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            batch_attention_masks.append(attention_mask)
            batch_e1_mask.append(e1_mask)
            batch_e2_mask.append(e2_mask)

            batch_e1_neg_input_ids.append(neg_input_ids_1)
            batch_e1_neg_token_type_ids.append(neg_token_type_ids_1)
            batch_e1_neg_attention_masks.append(neg_attention_mask_1)
            batch_e2_neg_input_ids.append(neg_input_ids_2)
            batch_e2_neg_token_type_ids.append(neg_token_type_ids_2)
            batch_e2_neg_attention_masks.append(neg_attention_mask_2)
            batch_e1_neg_mask.append(e1_neg_mask)
            batch_e2_neg_mask.append(e2_neg_mask)
            batch_ent.append({'e1':e1,'e2':e2})
            # batch_type.append(ent_type_res)
        if self.config.fixed_batch_length:
            pad_length = self.config.max_len
        else:
            pad_length = min(batch_max_len, self.config.max_len)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
                                            device=self.device).long()
        batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
                                             device=self.device).long()
        batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
        batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

        batch_labels = torch.tensor(batch_labels, device=self.device).long()

        batch_e1_neg_input_ids = torch.tensor(sequence_padding(batch_e1_neg_input_ids, length=self.neg_len), device=self.device).long()
        batch_e1_neg_token_type_ids = torch.tensor(sequence_padding(batch_e1_neg_token_type_ids, length=self.neg_len), device=self.device).long()
        batch_e1_neg_attention_masks = torch.tensor(sequence_padding(batch_e1_neg_attention_masks, length=self.neg_len), device=self.device).long()
        batch_e2_neg_input_ids = torch.tensor(sequence_padding(batch_e2_neg_input_ids, length=self.neg_len), device=self.device).long()
        batch_e2_neg_token_type_ids = torch.tensor(sequence_padding(batch_e2_neg_token_type_ids, length=self.neg_len), device=self.device).long()
        batch_e2_neg_attention_masks = torch.tensor(sequence_padding(batch_e2_neg_attention_masks, length=self.neg_len), device=self.device).long()
        batch_e1_neg_mask = torch.tensor(sequence_padding(batch_e1_neg_mask, length=self.neg_len), device=self.device).long()
        batch_e2_neg_mask = torch.tensor(sequence_padding(batch_e2_neg_mask, length=self.neg_len), device=self.device).long()
        # batch_type = torch.tensor(batch_type,device=self.device).long()
        if 'gcn' in self.config.model_name or 'gat' in self.config.model_name:
            return  batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels,batch_e1_neg_input_ids,batch_e1_neg_token_type_ids,batch_e1_neg_attention_masks,batch_e2_neg_input_ids,batch_e2_neg_token_type_ids,batch_e2_neg_attention_masks,batch_e1_neg_mask,batch_e2_neg_mask,batch_ent
            
        return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels,batch_e1_neg_input_ids,batch_e1_neg_token_type_ids,batch_e1_neg_attention_masks,batch_e2_neg_input_ids,batch_e2_neg_token_type_ids,batch_e2_neg_attention_masks,batch_e1_neg_mask,batch_e2_neg_mask

    def _process_seq_len(self, text, total_special_toks=3):
        """
            裁切句子的方法，直接使用clinicalTransformer提供的方法
            This function is used to truncate sequences with len > max_seq_len
            Truncate strategy:
            1. find all the index for special tags
            3. count distances between leading word to first tag and second tag to last.
            first -1- tag1 entity tag2 -2- last
            4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
            5. repeat until len is equal to max_seq_len
        """
        loop_count = 0
        while len(self.tokenizer.tokenize(text)) > (self.config.max_len - total_special_toks):
            text = self._truncate_helper(text)
            loop_count += 1
            if loop_count > 50:
                return
        return text

    def _truncate_helper(self, text):
        '''
        这是一个句子一个句子的找
        这里对原始的的text进行去除，并不是tokenize之后的....
        :param text:
        :return:
        '''
        tokens = text.split(" ")
        # 这是得到 word-level的index
        spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if
                                        tk.lower() in [self.config.ent1_start_tag, self.config.ent2_end_tag]]
        start_idx, end_idx = 0, len(tokens) - 1
        truncate_space_head = spec_tag_idx1 - start_idx
        truncate_space_tail = end_idx - spec_tag_idx2

        if truncate_space_head == truncate_space_tail == 0:  # 这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
            return text

        if truncate_space_head > truncate_space_tail:  # 如果离头更远，那么先抛弃头部的word...
            tokens.pop(0)
        else:
            tokens.pop(-1)

        return " ".join(tokens)

    
# class AbsNormalDataset(Dataset):
#     #这是针对摘要级别的
#     def __init__(self, examples, config: BertConfig, tokenizer, label2id, device):
#         super(AbsNormalDataset, self).__init__()
#         self.config = config
#         self.examples = examples

#         self.tokenizer = tokenizer
#         # 将特殊实体加入到分词器，防止给切分

#         self.label2id = label2id

#         self.max_len = config.max_len
#         self.device = device

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):

#         return self.examples[item]

#     def collate_fn_predicate(self, features):
#         '''
#         这个专用于模型的predicate的collate_fn
#         和collate_fn的不同是没有label的处理
#         :param features:
#         :return:
#         '''
#         raw_text_li = []
#         batch_input_ids = []
#         batch_token_type_ids = []
#         batch_attention_masks = []
#         batch_e1_mask = []
#         batch_e2_mask = []


#         batch_max_len = 0
#         for example in features:
#             sent = example.text
#             # sent 是word-level list: ['Feadeal','ABVDF','the',...]
#             raw_text_li.append(sent)
#             subword_tokens = self.tokenizer.tokenize(sent)
#             # 如果长度过长，那么开始裁剪长度
#             if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

#                 logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
#                 sent = self._process_seq_len(sent)
#                 if not sent:
#                     logger.warning('此数据难以裁剪，进行抛弃......')
#                     continue
#                 subword_tokens = self.tokenizer.tokenize(sent)
#                 logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
#                 if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
#                     continue

#             if batch_max_len < len(subword_tokens):
#                 batch_max_len = len(subword_tokens)

#             encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)

#             input_ids = encoder_res['input_ids']
#             token_type_ids = encoder_res['token_type_ids']
#             attention_mask = encoder_res['attention_mask']

#             e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
#             e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
#             e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
#             e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

#             e1_mask = np.zeros(len(input_ids))
#             e2_mask = np.zeros(len(input_ids))
#             e1_mask[e1_start_idx:e1_end_idx + 1] = 1
#             e2_mask[e2_start_idx:e2_end_idx + 1] = 1

#             batch_input_ids.append(input_ids)
#             batch_token_type_ids.append(token_type_ids)
#             batch_attention_masks.append(attention_mask)
#             batch_e1_mask.append(e1_mask)
#             batch_e2_mask.append(e2_mask)

#         if self.config.fixed_batch_length:
#             pad_length = self.config.max_len
#         else:
#             pad_length = min(batch_max_len, self.config.max_len)

#         batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
#         batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
#                                             device=self.device).long()
#         batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
#                                              device=self.device).long()
#         batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
#         batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

#         return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask

#     def collate_fn(self, features):
#         '''

#         :param features:
#         :return:
#         '''
#         raw_text_li = []
#         batch_input_ids = []
#         batch_token_type_ids = []
#         batch_labels = []
#         batch_attention_masks = []
#         batch_entities_positions = []
#         batch_e1_mask = []
#         batch_e2_mask = []
#         batch_e1_front_input_ids = []
#         batch_e1_front_token_type_ids = []
#         batch_e1_front_attention_masks = []
#         batch_e2_front_input_ids = []
#         batch_e2_front_token_type_ids = []
#         batch_e2_front_attention_masks = []
#         batch_e1_next_input_ids = []
#         batch_e1_next_token_type_ids = []
#         batch_e1_next_attention_masks = []
#         batch_e2_next_input_ids = []
#         batch_e2_next_token_type_ids = []
#         batch_e2_next_attention_masks = []

#         batch_max_len = 0
#         for example in features:

#             sent = example.text
#             label = self.label2id[example.label]
#             front1_text = example.ent1_front_sent
#             front2_text = example.ent2_front_sent
#             next1_text = example.ent1_next_sent
#             next2_text = example.ent1_next_sent

#             # sent 是word-level list: ['Feadeal','ABVDF','the',...]
#             raw_text_li.append(sent)
#             subword_tokens = self.tokenizer.tokenize(sent)
#             e1_front_subword_tokens = self.tokenizer.tokenize(front1_text)
#             e2_front_subword_tokens = self.tokenizer.tokenize(front2_text)
#             e1_next_subword_tokens = self.tokenizer.tokenize(next1_text)
#             e2_next_subword_tokens = self.tokenizer.tokenize(next2_text)
#             subword_tokens_list = [subword_tokens,e1_front_subword_tokens,e2_front_subword_tokens,e1_next_subword_tokens,e2_next_subword_tokens]
#             sent_list = [sent,front1_text,front2_text,next1_text,next2_text]
#             # 如果长度过长，那么开始裁剪长度
#             for  (i,subword) in enumerate(subword_tokens_list):
#                 if len(subword) > self.config.max_len - self.config.total_special_toks:

#                     logger.info('长度为{},开始裁剪长度'.format(len(subword)))
#                     sent_list[i] = self._process_seq_len(sent_list[i])
#                     if not sent_list[i]:
#                         logger.warning('此数据难以裁剪，进行抛弃......')
#                         continue
#                     subword_tokens_list[i] = self.tokenizer.tokenize(sent_list[i])
#                     logger.info('裁剪之后的长度为{}'.format(len(subword_tokens_list[i])))
#                     if len(subword_tokens_list[i]) > self.config.max_len - self.config.total_special_toks:
#                         continue
#             for i in subword_tokens_list:
#                 if batch_max_len < len(i):
#                     batch_max_len = len(i)
#             batch_labels.append(label)
#             subword_tokens,e1_front_subword_tokens,e2_front_subword_tokens,e1_next_subword_tokens,e2_next_subword_tokens = subword_tokens_list
#             encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)
#             e1_front_encoder_res = self.tokenizer.encode_plus(e1_front_subword_tokens,truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)
#             e2_front_encoder_res = self.tokenizer.encode_plus(e2_front_subword_tokens,truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)
#             e1_next_encoder_res = self.tokenizer.encode_plus(e1_next_subword_tokens,truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)
#             e2_next_encoder_res = self.tokenizer.encode_plus(e2_next_subword_tokens,truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)    

#             e1_front_input_ids = e1_front_encoder_res['input_ids']
#             e1_front_token_type_ids = e1_front_encoder_res['token_type_ids']
#             e1_front_attention_mask = e1_front_encoder_res['attention_mask']

#             e2_front_input_ids = e2_front_encoder_res['input_ids']
#             e2_front_token_type_ids = e2_front_encoder_res['token_type_ids']
#             e2_front_attention_mask = e2_front_encoder_res['attention_mask']

#             e1_next_input_ids = e1_next_encoder_res['input_ids']
#             e1_next_token_type_ids = e1_next_encoder_res['token_type_ids']
#             e1_next_attention_mask = e1_next_encoder_res['attention_mask']

#             e2_next_input_ids = e2_next_encoder_res['input_ids']
#             e2_next_token_type_ids = e2_next_encoder_res['token_type_ids']
#             e2_next_attention_mask = e2_next_encoder_res['attention_mask']

#             input_ids = encoder_res['input_ids']
#             token_type_ids = encoder_res['token_type_ids']
#             attention_mask = encoder_res['attention_mask']


#             e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
#             e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
#             e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
#             e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

#             e1_mask = np.zeros(len(input_ids))
#             e2_mask = np.zeros(len(input_ids))
#             e1_mask[e1_start_idx:e1_end_idx + 1] = 1
#             e2_mask[e2_start_idx:e2_end_idx + 1] = 1

#             batch_input_ids.append(input_ids)
#             batch_token_type_ids.append(token_type_ids)
#             batch_attention_masks.append(attention_mask)

#             batch_e1_front_input_ids.append(e1_front_input_ids)
#             batch_e1_front_token_type_ids.append(e1_front_token_type_ids)
#             batch_e1_front_attention_masks.append(e1_front_attention_mask)

#             batch_e2_front_input_ids.append(e2_front_input_ids)
#             batch_e2_front_token_type_ids.append(e2_front_token_type_ids)
#             batch_e2_front_attention_masks.append(e2_front_attention_mask)

#             batch_e1_next_input_ids.append(e1_next_input_ids)
#             batch_e1_next_token_type_ids.append(e1_next_token_type_ids)
#             batch_e1_next_attention_masks.append(e1_next_attention_mask)

#             batch_e2_next_input_ids.append(e2_next_input_ids)
#             batch_e2_next_token_type_ids.append(e2_next_token_type_ids)
#             batch_e2_next_attention_masks.append(e2_next_attention_mask)

#             batch_e1_mask.append(e1_mask)
#             batch_e2_mask.append(e2_mask)

#         if self.config.fixed_batch_length:
#             pad_length = self.config.max_len
#         else:
#             pad_length = min(batch_max_len, self.config.max_len)

#         batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
#         batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
#                                             device=self.device).long()
#         batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
#                                              device=self.device).long()
#         batch_e1_front_input_ids = torch.tensor(sequence_padding(batch_e1_front_input_ids,length=pad_length),device=self.device).long()
#         batch_e1_front_token_type_ids = torch.tensor(sequence_padding(batch_e1_front_token_type_ids,length=pad_length),device=self.device).long()
#         batch_e1_front_attention_masks = torch.tensor(sequence_padding(batch_e1_front_attention_masks,length=pad_length),device=self.device).long()

#         batch_e2_front_input_ids = torch.tensor(sequence_padding(batch_e2_front_input_ids,length=pad_length),device=self.device).long()
#         batch_e2_front_token_type_ids = torch.tensor(sequence_padding(batch_e2_front_token_type_ids,length=pad_length),device=self.device).long()
#         batch_e2_front_attention_masks = torch.tensor(sequence_padding(batch_e2_front_attention_masks,length=pad_length),device=self.device).long()

#         batch_e1_next_input_ids = torch.tensor(sequence_padding(batch_e1_next_input_ids,length=pad_length),device=self.device).long()
#         batch_e1_next_token_type_ids = torch.tensor(sequence_padding(batch_e1_next_token_type_ids,length=pad_length),device=self.device).long()
#         batch_e1_next_attention_masks = torch.tensor(sequence_padding(batch_e1_next_attention_masks,length=pad_length),device=self.device).long()

#         batch_e2_next_input_ids = torch.tensor(sequence_padding(batch_e2_next_input_ids,length=pad_length),device=self.device).long()
#         batch_e2_next_token_type_ids = torch.tensor(sequence_padding(batch_e2_next_token_type_ids,length=pad_length),device=self.device).long()
#         batch_e2_next_attention_masks = torch.tensor(sequence_padding(batch_e2_next_attention_masks,length=pad_length),device=self.device).long()


#         batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
#         batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

#         batch_labels = torch.tensor(batch_labels, device=self.device).long()

#         return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels,batch_e1_front_input_ids,batch_e1_front_token_type_ids,batch_e1_front_attention_masks,batch_e1_next_input_ids,batch_e1_next_token_type_ids,batch_e1_next_attention_masks,batch_e2_front_input_ids,batch_e2_front_token_type_ids,batch_e2_front_attention_masks,batch_e2_next_input_ids,batch_e2_next_token_type_ids,batch_e2_next_attention_masks

#     def _process_seq_len(self, text, total_special_toks=3):
#         """
#             裁切句子的方法，直接使用clinicalTransformer提供的方法
#             This function is used to truncate sequences with len > max_seq_len
#             Truncate strategy:
#             1. find all the index for special tags
#             3. count distances between leading word to first tag and second tag to last.
#             first -1- tag1 entity tag2 -2- last
#             4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
#             5. repeat until len is equal to max_seq_len
#         """
#         loop_count = 0
#         while len(self.tokenizer.tokenize(text)) > (self.config.max_len - total_special_toks):
#             text = self._truncate_helper(text)
#             loop_count += 1
#             if loop_count > 50:
#                 return
#         return text

#     def _truncate_helper(self, text):
#         '''
#         这是一个句子一个句子的找
#         这里对原始的的text进行去除，并不是tokenize之后的....
#         :param text:
#         :return:
#         '''
#         tokens = text.split(" ")
#         # 这是得到 word-level的index
#         spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if
#                                         tk.lower() in [self.config.ent1_start_tag, self.config.ent2_end_tag]]
#         start_idx, end_idx = 0, len(tokens) - 1
#         truncate_space_head = spec_tag_idx1 - start_idx
#         truncate_space_tail = end_idx - spec_tag_idx2

#         if truncate_space_head == truncate_space_tail == 0:  # 这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
#             return text

#         if truncate_space_head > truncate_space_tail:  # 如果离头更远，那么先抛弃头部的word...
#             tokens.pop(0)
#         else:
#             tokens.pop(-1)

#         return " ".join(tokens)
    
# class ALLAbsNormalDataset(Dataset):
#     def __init__(self, examples, config: BertConfig, tokenizer, label2id, device):
#         super(ALLAbsNormalDataset, self).__init__()
#         self.config = config
#         self.examples = examples

#         self.tokenizer = tokenizer
#         # 将特殊实体加入到分词器，防止给切分

#         self.label2id = label2id
#         self.sent_max_len =config.sent_max_len
#         self.max_len = config.max_len
#         self.device = device

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):

#         return self.examples[item]

#     def collate_fn_predicate(self, features):
#         '''
#         这个专用于模型的predicate的collate_fn
#         和collate_fn的不同是没有label的处理
#         :param features:
#         :return:
#         '''
#         raw_text_li = []
#         batch_input_ids = []
#         batch_token_type_ids = []
#         batch_attention_masks = []
#         batch_e1_mask = []
#         batch_e2_mask = []



#         batch_max_len = 0
#         for example in features:
#             sent = example.text
#             # sent 是word-level list: ['Feadeal','ABVDF','the',...]
#             raw_text_li.append(sent)
#             subword_tokens = self.tokenizer.tokenize(sent)
#             # 如果长度过长，那么开始裁剪长度
#             if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:

#                 logger.info('长度为{},开始裁剪长度'.format(len(subword_tokens)))
#                 sent = self._process_seq_len(sent)
#                 if not sent:
#                     logger.warning('此数据难以裁剪，进行抛弃......')
#                     continue
#                 subword_tokens = self.tokenizer.tokenize(sent)
#                 logger.info('裁剪之后的长度为{}'.format(len(subword_tokens)))
#                 if len(subword_tokens) > self.config.max_len - self.config.total_special_toks:
#                     continue

#             if batch_max_len < len(subword_tokens):
#                 batch_max_len = len(subword_tokens)

#             encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)

#             input_ids = encoder_res['input_ids']
#             token_type_ids = encoder_res['token_type_ids']
#             attention_mask = encoder_res['attention_mask']

#             e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
#             e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
#             e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
#             e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

#             e1_mask = np.zeros(len(input_ids))
#             e2_mask = np.zeros(len(input_ids))
#             e1_mask[e1_start_idx:e1_end_idx + 1] = 1
#             e2_mask[e2_start_idx:e2_end_idx + 1] = 1

#             batch_input_ids.append(input_ids)
#             batch_token_type_ids.append(token_type_ids)
#             batch_attention_masks.append(attention_mask)
#             batch_e1_mask.append(e1_mask)
#             batch_e2_mask.append(e2_mask)

#         if self.config.fixed_batch_length:
#             pad_length = self.config.max_len
#         else:
#             pad_length = min(batch_max_len, self.config.max_len)

#         batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
#         batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
#                                             device=self.device).long()
#         batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
#                                              device=self.device).long()
#         batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
#         batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

#         return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask

#     def collate_fn(self, features):
#         '''

#         :param features:
#         :return:
#         '''
#         raw_text_li = []
#         batch_input_ids = []
#         batch_token_type_ids = []
#         batch_labels = []
#         batch_attention_masks = []
#         batch_entities_positions = []
#         batch_e1_mask = []
#         batch_e2_mask = []

#         batch_abs_input_ids = []
#         batch_abs_token_type_ids = []
#         batch_abs_attention_masks = []

#         abs_length = []

#         batch_max_len = 0
#         for example in features:
#             abs_subword_tokens = []
#             abstract = example.abstract
#             sent = example.text
#             label = self.label2id[example.label]
#             all_sent= sent_tokenize(abstract)
#             #记录每个abs句子的个数
#             abs_length.append(len(all_sent))

#             # sent 是word-level list: ['Feadeal','ABVDF','the',...]
#             raw_text_li.append(sent)
#             raw_all_text = []
#             #将每个句子加入到列表中
#             for i in all_sent:
#                 raw_all_text.append(i)
#                 abs_subword_tokens.append(self.tokenizer.tokenize(i))

#             subword_tokens = self.tokenizer.tokenize(sent)
#             # 如果长度过长，那么开始裁剪长度
#             new_subword_tokens = [subword_tokens] + abs_subword_tokens
#             new_raw_all_text = [sent] + raw_all_text

#             for (i,j) in enumerate(new_subword_tokens):
#                 if len(j) > self.config.max_len - self.config.total_special_toks:

#                     logger.info('长度为{},开始裁剪长度'.format(len(j)))
#                     new_raw_all_text[i] = self._process_seq_len(new_raw_all_text[i])
#                     if not new_raw_all_text[i]:
#                         logger.warning('此数据难以裁剪，进行抛弃......')
#                         continue
#                     new_subword_tokens[i] = self.tokenizer.tokenize(new_raw_all_text[i])
#                     logger.info('裁剪之后的长度为{}'.format(len(new_subword_tokens[i])))
#                     if len(new_subword_tokens[i]) > self.config.max_len - self.config.total_special_toks:
#                         continue
#             for i in new_subword_tokens:
#                 if batch_max_len < len(i):
#                     batch_max_len = len(i)
#             batch_labels.append(label)

#             subword_tokens = new_subword_tokens[0]
#             abs_subword_tokens = new_subword_tokens[1:]
#             for i in abs_subword_tokens:
#                 encoder_res = self.tokenizer.encode_plus(i, truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)
#                 input_ids = encoder_res['input_ids']
#                 token_type_ids = encoder_res['token_type_ids']
#                 attention_mask = encoder_res['attention_mask']
#                 batch_abs_input_ids.append(input_ids)
#                 batch_abs_token_type_ids.append(token_type_ids)
#                 batch_abs_attention_masks.append(attention_mask)


#             encoder_res = self.tokenizer.encode_plus(subword_tokens, truncation=True, max_length=self.max_len,
#                                                      add_special_tokens=True)

#             input_ids = encoder_res['input_ids']
#             token_type_ids = encoder_res['token_type_ids']
#             attention_mask = encoder_res['attention_mask']


#             e1_start_idx = input_ids.index(self.config.ent1_start_tag_id)
#             e1_end_idx = input_ids.index(self.config.ent1_end_tag_id)
#             e2_start_idx = input_ids.index(self.config.ent2_start_tag_id)
#             e2_end_idx = input_ids.index(self.config.ent2_end_tag_id)

#             e1_mask = np.zeros(len(input_ids))
#             e2_mask = np.zeros(len(input_ids))
#             e1_mask[e1_start_idx:e1_end_idx + 1] = 1
#             e2_mask[e2_start_idx:e2_end_idx + 1] = 1

#             batch_input_ids.append(input_ids)
#             batch_token_type_ids.append(token_type_ids)
#             batch_attention_masks.append(attention_mask)
#             batch_e1_mask.append(e1_mask)
#             batch_e2_mask.append(e2_mask)

#         if self.config.fixed_batch_length:
#             pad_length = self.config.max_len
#         else:
#             pad_length = min(batch_max_len, self.config.max_len)

#         batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=pad_length), device=self.device).long()
#         batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=pad_length),
#                                             device=self.device).long()
#         batch_attention_masks = torch.tensor(sequence_padding(batch_attention_masks, length=pad_length),
#                                              device=self.device).long()
#         batch_e1_mask = torch.tensor(sequence_padding(batch_e1_mask, length=pad_length), device=self.device).long()
#         batch_e2_mask = torch.tensor(sequence_padding(batch_e2_mask, length=pad_length), device=self.device).long()

#         batch_abs_input_ids = sequence_padding(batch_abs_input_ids, length=pad_length)
#         batch_abs_token_type_ids = sequence_padding(batch_abs_token_type_ids, length=pad_length)
#         batch_abs_attention_masks = sequence_padding(batch_abs_attention_masks, length=pad_length)

#         index = 0
#         new_batch_abs_input_ids = []
#         new_batch_abs_token_type_ids = []
#         new_batch_abs_attention_masks = []
#         for (i,length) in enumerate(abs_length):
#             abs_input_ids = batch_abs_input_ids[index:index+length]
#             abs_token_type_ids = batch_abs_token_type_ids[index:index+length]
#             abs_attention_masks = batch_abs_attention_masks[index:index+length]
#             new_batch_abs_input_ids.append(abs_input_ids)
#             new_batch_abs_token_type_ids.append(abs_token_type_ids)
#             new_batch_abs_attention_masks.append(abs_attention_masks)
#             index += length

#         new_batch_abs_input_ids = torch.tensor(sequence_padding(new_batch_abs_input_ids,length=self.sent_max_len),device=self.device).long()
#         new_batch_abs_token_type_ids = torch.tensor(sequence_padding(new_batch_abs_token_type_ids,length=self.sent_max_len),device=self.device).long()
#         new_batch_abs_attention_masks = torch.tensor(sequence_padding(new_batch_abs_attention_masks,length=self.sent_max_len),device=self.device).long()

#         batch_labels = torch.tensor(batch_labels, device=self.device).long()

#         return batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_e1_mask, batch_e2_mask, batch_labels,new_batch_abs_input_ids,new_batch_abs_token_type_ids,new_batch_abs_attention_masks

#     def _process_seq_len(self, text, total_special_toks=3):
#         """
#             裁切句子的方法，直接使用clinicalTransformer提供的方法
#             This function is used to truncate sequences with len > max_seq_len
#             Truncate strategy:
#             1. find all the index for special tags
#             3. count distances between leading word to first tag and second tag to last.
#             first -1- tag1 entity tag2 -2- last
#             4. pick the longest distance from (1, 2), if 1 remove first token, if 2 remove last token
#             5. repeat until len is equal to max_seq_len
#         """
#         loop_count = 0
#         while len(self.tokenizer.tokenize(text)) > (self.config.max_len - total_special_toks):
#             text = self._truncate_helper(text)
#             loop_count += 1
#             if loop_count > 50:
#                 return
#         return text

#     def _truncate_helper(self, text):
#         '''
#         这是一个句子一个句子的找
#         这里对原始的的text进行去除，并不是tokenize之后的....
#         :param text:
#         :return:
#         '''
#         tokens = text.split(" ")
#         # 这是得到 word-level的index
#         spec_tag_idx1, spec_tag_idx2 = [idx for (idx, tk) in enumerate(tokens) if
#                                         tk.lower() in [self.config.ent1_start_tag, self.config.ent2_end_tag]]
#         start_idx, end_idx = 0, len(tokens) - 1
#         truncate_space_head = spec_tag_idx1 - start_idx
#         truncate_space_tail = end_idx - spec_tag_idx2

#         if truncate_space_head == truncate_space_tail == 0:  # 这是表示如果实体1和实体2 都已经在句子的首尾，那么就不要继续删除了....
#             return text

#         if truncate_space_head > truncate_space_tail:  # 如果离头更远，那么先抛弃头部的word...
#             tokens.pop(0)
#         else:
#             tokens.pop(-1)

#         return " ".join(tokens)


