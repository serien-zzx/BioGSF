# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  数据读取
   Author :        kedaxia
   date：          2021/12/02
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/02: 
-------------------------------------------------
"""
import random

from ipdb import set_trace

import numpy as np

from gensim.models import Word2Vec, FastText

import torch

from torch_geometric.utils import dense_to_sparse

class InputExamples(object):
    def __init__(self, text, label, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id=None, ent2_id=None,
                 abstract_id=None, rel_type=None):
        '''
        针对sentence-level的关系分类任务....
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        '''
        self.text = text
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.abstract_id = abstract_id
        self.rel_type = rel_type


class MTBExamples(object):
    def __init__(self, text_a, text_b, label, ent1_type, ent2_type, ent1_name=None, ent2_name=None, ent1_id=None,
                 ent2_id=None, abstract_id=None, rel_type=None):
        '''
        MTB的cross-sentence 关系分类任务
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        :param ent1_name:
        :param ent2_name:
        '''
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.abstract_id = abstract_id
        self.rel_type = rel_type

class NEG_InputExamples(object):
    def __init__(self, text ,ent1_type, ent2_type, ent1_name, ent2_name, label,e1_neg,e2_neg,rel_type,ent1_id=None, ent2_id=None,
                 abstract_id=None):
        '''
        针对sentence-level的关系分类任务....
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        '''
        self.text = text
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.e1_neg = e1_neg
        self.e2_neg = e2_neg
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.abstract_id = abstract_id
        self.rel_type = rel_type

class ALLNEG_InputExamples(object):
    def __init__(self, text ,ent1_type, ent2_type, ent1_name, ent2_name, label,neg_sent,rel_type):
        '''
        针对sentence-level的关系分类任务....
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        '''
        self.text = text
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.neg_sent = neg_sent


        self.rel_type = rel_type


class ABS_InputExamples(object):
    def __init__(self, text ,ent1_type, ent2_type, ent1_name, ent2_name, label, front1_text,next1_text,front2_text,next2_text,rel_type):
        '''
        针对sentence-level的关系分类任务....
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        '''
        self.text = text
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.ent1_front_sent = front1_text
        self.ent2_front_sent = front2_text
        self.ent1_next_sent = next1_text
        self.ent2_next_sent = next2_text
        self.rel_type = rel_type

class ALLABS_InputExamples(object):
    def __init__(self, text ,ent1_type, ent2_type, ent1_name, ent2_name, label,abstract,rel_type):
        '''
        针对sentence-level的关系分类任务....
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        '''
        self.text = text
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.abstract = abstract
        self.rel_type = rel_type


def load_pretrained_fasttext(fastText_embedding_path):
    '''
    加载预训练的fastText
    :param fastText_embedding_path:
    :return:fasttext,word2id,id2word
    '''
    fasttext = FastText.load(fastText_embedding_path)

    id2word = {i + 1: j for i, j in enumerate(fasttext.wv.index2word)}  # 共1056283个单词，也就是这些embedding
    word2id = {j: i for i, j in id2word.items()}
    fasttext = fasttext.wv.syn0
    word_hidden_dim = fasttext.shape[1]
    # 这是为了unk
    fasttext = np.concatenate([np.zeros((1, word_hidden_dim)), np.zeros((1, word_hidden_dim)), fasttext])
    return fasttext, word2id, id2word


def load_pretrained_word2vec(word2vec_embedding_path):
    '''
    加载预训练的fastText
    :param word2vec_embedding_path:
    :return:word2vec, word2id, id2word
    '''
    word2vec = Word2Vec.load(word2vec_embedding_path)

    # 空出0和1，0是pad，1是unknow
    id2word = {i + 2: j for i, j in enumerate(word2vec.wv.index2word)}  # 共1056283个单词，也就是这些embedding
    word2id = {j: i for i, j in id2word.items()}
    word2vec = word2vec.wv.syn0
    word_hidden_dim = word2vec.shape[1]
    # 这是为了pad和unk
    word2id['unk'] = 1
    word2id['pad'] = 0
    id2word[0] = 'pad'
    id2word[1] = 'unk'
    word2vec = np.concatenate([np.zeros((1, word_hidden_dim)), np.zeros((1, word_hidden_dim)), word2vec])

    # word2vec = np.concatenate([[copy.deepcopy(word2vec[0])], word2vec])

    return word2vec, word2id, id2word


def get_relative_pos_feature(x, limit):
    """
       :param x = idx - entity_idx
       这个方法就是不管len(sentence)多长，都限制到这个位置范围之内

       x的范围就是[-len(sentence),len(sentence)] 转换到都是正值范围
       -limit ~ limit => 0 ~ limit * 2+2
       将范围转换一下，为啥
   """
    if x < -limit:
        return 0
    elif x >= -limit and x <= limit:
        return x + limit + 1
    else:
        return limit * 2 + 2


def get_label2id(label_file):
    f = open(label_file, 'r')
    t = f.readlines()
    f.close()
    label2id = {}
    id2label = {}
    for i, label in enumerate(t):
        label = label.strip()
        label2id[label] = i
        id2label[i] = label

    return label2id, id2label


def read_semeval2010(sentences_file, labels_file):
    '''
        关系分类任务，一般是读取两个文件，sentence.txt labels.txt
        这里就是读取数据
        :param file_path:
        :return:
            sents:列表，每一个为元组(start_idx,end_idx,entity_name)
            labels:对应的关系类别
        '''
    sents = list()
    # Replace each token by its index if it is in vocab, else use index of unk_word
    with open(sentences_file, 'r') as f:
        for i, line in enumerate(f):
            # 这里分离出实体对和句子
            e1, e2, sent = line.strip().split('\t')
            words = sent.split(' ')  # 将句子划分为一个一个单词
            sents.append((e1, e2, words))

    # Replace each label by its index
    f = open(labels_file, 'r')
    labels = f.readlines()
    f.close()
    labels = [label.strip() for label in labels]

    return sents, labels


def read_file(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    t = f.readlines()
    f.close()
    res = [x.strip() for x in t]
    return res


def read_raw_data(config):
    '''
    这里根据不同的数据集，需要读取不同格式的数据集，但是最后输出会保持一致，一个是sentence，另一个是label
    :param config:
    :param type:
    :return:
    '''

    if config.data_format == 'single':  # 格式为<CLS>sentence a<sep>sentence b <sep>
        examples = process_raw_normal_data(config.dev_normal_path,config.dataset_name)

    elif config.data_format == 'cross':
        examples = process_raw_mtb_data(config.dev_mtb_path)
    elif config.data_format == 'neg_single':
        examples = process_raw_neg_normal_data(config.dev_normal_path, config.dataset_name)

    else:
        raise ValueError("data_format错误")
    return examples


def read_data(config, type_='train'):
    '''
    这里根据不同的数据集，需要读取不同格式的数据集，但是最后输出会保持一致，一个是sentence，另一个是label
    :param config:
    :param type:
    :return:
    '''

    if config.dataset_name == 'semeval2010':
        if type_ == 'train':
            return read_semeval2010(config.train_file_path, config.train_labels_path)
        else:
            return read_semeval2010(config.dev_file_path, config.dev_labels_path)
    else:
        if config.data_format == 'single':  # 格式为<CLS>sentence a<sep>sentence b <sep>
            if type_ == 'train':
                examples = process_normal_data(config.train_normal_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_normal_data(config.dev_normal_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_normal_data(config.test_normal_path, config.dataset_name)
        elif config.data_format == 'abs_single':
            if type_ == 'train':
                examples = process_abs_normal_data(config.train_normal_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_abs_normal_data(config.dev_normal_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_abs_normal_data(config.test_normal_path, config.dataset_name)
        elif config.data_format =='all_abs_single':
            if type_ == 'train':
                examples = process_all_abs_normal_data(config.train_normal_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_all_abs_normal_data(config.dev_normal_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_all_abs_normal_data(config.test_normal_path, config.dataset_name)
        elif config.data_format == 'cross':
            if type_ == 'train':
                examples = process_mtb_data(config.train_mtb_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_mtb_data(config.dev_mtb_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_mtb_data(config.test_mtb_path, config.dataset_name)
        elif config.data_format == 'inter':
            if type_ == 'train':
                examples = process_mtb_data(config.train_mtb_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_mtb_data(config.dev_mtb_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_mtb_data(config.test_mtb_path, config.dataset_name)
        elif config.data_format == 'neg_single':
            if type_ == 'train':
                examples = process_neg_normal_data(config.train_normal_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_neg_normal_data(config.dev_normal_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_neg_normal_data(config.test_normal_path, config.dataset_name)
        elif config.data_format == 'all_neg_single':
            if type_ == 'train':
                examples = process_all_neg_normal_data(config.train_normal_path, config.dataset_name)
            elif type_ == 'dev':
                examples = process_all_neg_normal_data(config.dev_normal_path, config.dataset_name)
            elif type_ == 'test':
                examples = process_all_neg_normal_data(config.test_normal_path, config.dataset_name)
        else:
            raise ValueError("data_format value error， please choise ['single','cross']")
        return examples


def process_mtb_data(file_path, dataset_name):
    '''

    :param file_path:
    :return:
    '''
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []

    if dataset_name in ['DDI2013', 'LLL', 'HPRD-50', 'IEPA', 'AIMed','BioInfer']:  # 针对二分类数据
        for idx, line in enumerate(lines):
            line = line.strip()
            line = line.split('\t')

            sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type, label = line
            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name)
            res.append(example)
    elif dataset_name in ['BC6ChemProt', 'BC7DrugProt']:
        for idx, line in enumerate(lines[1:]):

            line = line.strip()  # 去除换行符
            line = line.split('\t')
            if dataset_name == 'BC6ChemProt':
                sent1, sent2, ent1_type, ent2_type, ent1_name, ent2_name, label, _, _, _ = line
                label2rel={
                    'CPR:1':1,
                    'CPR:2':2,
                    'CPR:3':3,
                    'CPR:4':4,
                    'CPR:5':5,
                    'CPR:6':6,
                    'CPR:7':7,
                    'CPR:8':8,
                    'CPR:9':9,
                    'CPR:10':10,
                }
            else:
                sent1, sent2, ent1_type, ent2_type, ent1_name, ent2_name, label, _, _ = line
                label2rel = {
                    'INHIBITOR': 1,
                    'PART-OF': 2,
                    'SUBSTRATE': 3,
                    'ACTIVATOR': 4,
                    'INDIRECT-DOWNREGULATOR': 5,
                    'ANTAGONIST': 6,
                    'INDIRECT-UPREGULATOR': 7,
                    'AGONIST': 8,
                    'DIRECT-REGULATOR': 9,
                    'PRODUCT-OF': 10,
                    'AGONIST-ACTIVATOR': 11,
                    'AGONIST-INHIBITOR': 12,
                    'SUBSTRATE_PRODUCT-OF': 130,

                }


            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=label2rel[label])
            res.append(example)
    elif dataset_name in ['BC5CDR', 'two_BC6', 'two_BC7']:
        for idx, line in enumerate(lines):
            line = line.strip()  # 去除换行符
            line = line.split('\t')
            sent1, sent2, ent1_type, ent2_type, ent1_name, ent2_name, label, _ = line
            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name)
            res.append(example)
    elif dataset_name == 'AllDataset':
        for idx, line in enumerate(lines[1:]):
            line = line[:-1]  # 去除换行符
            line = line.split('\t')

            sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type, label, _ = line
            label2rel = {
                '0': 1,
                '1': 2,
                '2': 3,
                '3': 4,
                '4': 5,
                '5': 6,
            }
            example = MTBExamples(sent1, sent2, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=label2rel[label])
            res.append(example)
    else:
        raise ValueError("选择正确的数据集名称")
    return res

def process_abs_normal_data(file_path, dataset_name):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name in ['BC5CDR','BC6ChemProt','BC7DrugProt']:
        for line in lines:
            line = line.strip()
            sent, ent1_type, ent2_type, ent1_name, ent2_name, label, front1_text,next1_text,front2_text,next2_text = line.split('\t')
            example = ABS_InputExamples(sent,ent1_type, ent2_type, ent1_name, ent2_name, label, front1_text,next1_text,front2_text,next2_text,rel_type=0)
            res.append(example)

    return res

def process_neg_normal_data(file_path, dataset_name):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name in ['DDI2013','BC6ChemProt','BC7DrugProt','GAD','euadr','LLL', 'HPRD-50', 'IEPA', 'AIMed','BioInfer','DiMeX','BC5CDR','Dataset_lite','BIORED','ChemProt']:
        for line in lines:
            line = line.strip()
            sent, ent1_type, ent2_type, ent1_name, ent2_name, label,e1_neg,e2_neg= line.split('\t')
            example = NEG_InputExamples(sent,ent1_type, ent2_type, ent1_name, ent2_name, label,e1_neg,e2_neg,rel_type=0)
            res.append(example)
    elif dataset_name in ['Dataset_plus','Dataset_plus_cv']:
        for line in lines:
            line = line.strip()
            sent, ent1_type, ent2_type, ent1_name, ent2_name, label,e1_neg,e2_neg= line.split('\t')
            if (ent1_type,ent2_type) in [("protein","protein")]:
                rel_type = 1
            elif (ent1_type,ent2_type) in [("drug","drug")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [('CHEMICAL','protein'),("Chemical/Drug","Gene/Protein")]:
                rel_type = 3

            elif (ent1_type, ent2_type) in [("Chemical","Disease"),("Disease","Chemical")]:
                rel_type = 4
            else:
                raise ValueError
            example = NEG_InputExamples(sent,ent1_type, ent2_type, ent1_name, ent2_name, label,e1_neg,e2_neg,rel_type=rel_type)
            res.append(example)
    elif dataset_name in ['DMI','CDI','PPI','CPI']:
        for line in lines:
            line = line.strip()
            sent,  ent1_name, ent2_name, ent1_type, ent2_type, label,e1_neg,e2_neg= line.split('\t')
            example = NEG_InputExamples(sent,ent1_type, ent2_type, ent1_name, ent2_name, label,e1_neg,e2_neg,rel_type=0)
            res.append(example)
    return res

def process_all_neg_normal_data(file_path, dataset_name):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name in ['DDI2013','BC6ChemProt','BC7DrugProt','Dataset_lite']:
        for line in lines:
            line = line.strip()
            sent, ent1_type, ent2_type, ent1_name, ent2_name, label,neg_sent= line.split('\t')
            example = ALLNEG_InputExamples(sent,ent1_type, ent2_type, ent1_name, ent2_name, label,neg_sent,rel_type=0)
            res.append(example)
    return res

def process_all_abs_normal_data(file_path, dataset_name):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name in ['BC5CDR','BC6ChemProt','BC7DrugProt']:
        for line in lines:
            line = line.strip()
            sent, ent1_type, ent2_type, ent1_name, ent2_name, label, abstract = line.split('\t')
            example = ALLABS_InputExamples(sent,ent1_type, ent2_type, ent1_name, ent2_name, label,abstract,rel_type=0)
            res.append(example)

    return res

def process_raw_mtb_data(file_path):
    '''

    :param file_path:
    :return:
    '''
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    for idx, line in enumerate(lines[1:]):
        line = line[:-1]  # 去除换行符
        line = line.split('\t')
        abstract_id, sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line
        example = MTBExamples(sent1, sent2, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id)
        res.append(example)
    return res


def process_raw_normal_data(file_path, dataset_name):
    """
    这是处理predicate所需要的raw dataset
    """
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name in ['Case_study']:
        for idx, line in enumerate(lines[:]):
            line = line.strip()
            line = line.split('\t')

            abstract_id, sent, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line

            if (ent1_type,ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
                rel_type = 0 

            elif (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
                rel_type = 1

            elif (ent1_type,ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [("mutation","Disease"),("Disease","mutation")]:
                rel_type = 3

            elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
                rel_type = 4

            elif (ent1_type, ent2_type) in [("Chemical/Drug","mutation"),("mutation","Chemical/Drug")]:
                rel_type = 5

            elif (ent1_type, ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
                rel_type = 6
            else:
                raise ValueError

            example = InputExamples(sent, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id,
                                    abstract_id=abstract_id,rel_type=rel_type)
            res.append(example)
    elif dataset_name in ['Case_study_1']:
        for idx, line in enumerate(lines[:]):
            line = line.strip()
            line = line.split('\t')
            abstract_id, sent, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line

            if (ent1_type,ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
                rel_type = 0 

            elif (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
                rel_type = 1

            elif (ent1_type,ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
                rel_type = 3

            elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
                rel_type = 4
                
            example = InputExamples(sent, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id,
                                    abstract_id=abstract_id,rel_type=rel_type)
            res.append(example)

    else:
        for idx, line in enumerate(lines[1:]):
            line = line.strip()
            line = line.split('\t')

            abstract_id, sent, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line

            if (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
                rel_type = 1
            elif (ent1_type,ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
                rel_type = 3

            elif (ent1_type, ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
                rel_type = 4
            elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
                rel_type = 5
            elif (ent1_type, ent2_type) in [("mutation","Gene/Protein"),("Gene/Protein","mutation")]:
                rel_type = 6
            elif (ent1_type, ent2_type) in [("mutation","Disease"),("Disease","mutation")]:
                rel_type = 7
            else:
                raise ValueError

            example = InputExamples(sent, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id,
                                    abstract_id=abstract_id,rel_type=rel_type)
            res.append(example)

    return res

def process_raw_neg_normal_data(file_path, dataset_name):
    """
    这是处理predicate所需要的raw dataset
    """
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name == 'Case_study':
        for idx, line in enumerate(lines[:]):
            line = line.strip()
            abstract_id, sent,ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, e1_neg,e2_neg= line.split('\t')


            if (ent1_type,ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
                rel_type = 0 

            elif (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
                rel_type = 1

            elif (ent1_type,ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [("mutation","Disease"),("Disease","mutation")]:
                rel_type = 3

            elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
                rel_type = 4

            elif (ent1_type, ent2_type) in [("Chemical/Drug","mutation"),("mutation","Chemical/Drug")]:
                rel_type = 5

            elif (ent1_type, ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
                rel_type = 6
            else:
                raise ValueError

            example = NEG_InputExamples(sent, ent1_type, ent2_type, ent1_name, ent2_name, None,e1_neg,e2_neg, rel_type,ent1_id, ent2_id,
                                    abstract_id=abstract_id)
            res.append(example)
    else:
        for idx, line in enumerate(lines[:]):
            line = line.strip()
            abstract_id, sent,ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, e1_neg,e2_neg= line.split('\t')
            if (ent1_type,ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
                rel_type = 0 

            elif (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
                rel_type = 1

            elif (ent1_type,ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
                rel_type = 3

            elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
                rel_type = 4

            example = NEG_InputExamples(sent, ent1_type, ent2_type, ent1_name, ent2_name, None,e1_neg,e2_neg, rel_type,ent1_id, ent2_id,
                                    abstract_id=abstract_id)
            res.append(example)
    return res

def process_normal_data(file_path, dataset_name):
    """
    这是处理标准数据集，数据格式为normal格式
    :param file_path:
    :return:
    """

    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    if dataset_name in ['2018n2c2_track2']:
        for line in lines:
            line = line.strip()
            rel_type, text_a, text_b, ent1_type, ent2_type, ent1_id, ent_id, _ = line.split('\t')
            example = MTBExamples(text_a, text_b, rel_type, ent1_type, ent2_type)
            res.append(example)

    elif dataset_name in ['Dataset_lite','Dataset_plus','BIORED','ChemProt']:
        for idx, line in enumerate(lines[:]):

            line = line.strip()
            line = line.split('\t')

            sent, ent1_name, ent2_name, ent1_type, ent2_type, label,_ = line

            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name, rel_type=0)
            res.append(example)

    elif dataset_name in ['euadr', 'GAD', 'DDI2013', 'LLL', 'HPRD-50', 'IEPA', 'AIMed','BioInfer','GDI','DMI']:  # 针对二分类数据
        for idx, line in enumerate(lines):

            line = line.strip() 
            line = line.split('\t')
            if dataset_name in ['euadr', 'GAD','GDI']:
                line = line[1:]
            if dataset_name == 'CPI':
                sent, ent1_type, ent2_type, ent1_name, ent2_name, label, _ = line
            else:
                sent, ent1_name, ent2_name, ent1_type, ent2_type, label = line


            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name, rel_type=0)
            res.append(example)

    elif dataset_name in ['DiMeX']:
        for idx, line in enumerate(lines[:]):

            line = line.strip()
            line = line.split('\t')

            sent, ent1_type, ent2_type, ent1_name, ent2_name, label= line

            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name, rel_type=0)
            res.append(example)

    elif dataset_name in ['BC6ChemProt']:
        for idx, line in enumerate(lines[1:]):

            line = line.strip()
            line = line.split('\t')

            sent, ent1_type, ent2_type, ent1_name, ent2_name, label, _ ,_,_= line

            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name, rel_type=0)
            res.append(example)
    elif dataset_name in ['BC5CDR', 'two_BC6', 'two_BC7','CDI', 'BC7DrugProt','PPI','CPI']:
        for idx, line in enumerate(lines):
            line = line.strip()
            line = line.split('\t')
            sent, ent1_type, ent2_type, ent1_name, ent2_name, label, _ = line

            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=0)
            res.append(example)
    elif dataset_name == 'AllDataset' or 'CV' in dataset_name:
        for idx, line in enumerate(lines[1:]):
            line = line.strip()
            line = line.split('\t')
            sent, ent1_name, ent2_name, ent1_type, ent2_type, label, _ = line

            if ent1_name == ent2_name:
                continue
            # if (ent1_type,ent2_type) in [("Gene/Protein","Gene/Protein"),('DNA','Gene/Protein'),('Gene/Protein','DNA'),('RNA','Gene/Protein'),('Gene/Protein','RNA'),('RNA','RNA'),('DNA','DNA')]:
            #     rel_type = 1
            # elif (ent1_type,ent2_type) in [("Chemical/Drug","Chemical/Drug")]:
            #     rel_type = 2
            #
            # elif (ent1_type, ent2_type) in [('Gene/Protein','Chemical/Drug'),("Chemical/Drug","Gene/Protein")]:
            #     rel_type = 3
            # elif (ent1_type, ent2_type) in [("Gene/Protein","Disease"),("Disease","Gene/Protein")]:
            #     rel_type = 4
            # elif (ent1_type, ent2_type) in [("Chemical/Drug","Disease"),("Disease","Chemical/Drug")]:
            #     rel_type = 5
            if (ent1_type,ent2_type) in [("protein","protein")]:
                rel_type = 1
            elif (ent1_type,ent2_type) in [("drug","drug")]:
                rel_type = 2

            elif (ent1_type, ent2_type) in [('CHEMICAL','protein'),("protein","CHEMICAL")]:
                rel_type = 3
            elif (ent1_type, ent2_type) in [("GENE","DISEASE"),("DISEASE","GENE")]:
                rel_type = 4
            elif (ent1_type, ent2_type) in [("Chemical","Disease"),("Disease","Chemical")]:
                rel_type = 5
            elif (ent1_type, ent2_type) in [("mutation","GENE"),("GENE","mutation")]:
                rel_type = 6
            elif (ent1_type, ent2_type) in [("mutation","DISEASE"),("DISEASE","mutation")]:
                rel_type = 7
            else:
                print(ent1_type,ent2_type)
                raise ValueError


            example = InputExamples(sent, label, ent1_type, ent2_type, ent1_name, ent2_name,rel_type=rel_type)
            res.append(example)


    else:
        raise ValueError("选择正确的数据集名称")
    return res


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    '''
    这里对数据进行pad，不同的batch里面使用不同的长度
    这个方法从多个方面考虑pad，写的很高级
    这个方法一般写不出来，阿西吧


    Numpy函数，将序列padding到同一长度
    按照一个batch的最大长度进行padding
    :param inputs:(batch_size,None),每个序列的长度不一样
    :param seq_dim: 表示对哪些维度进行pad，默认为1，只有当对label进行pad的时候，seq_dim=3,因为labels.shape=(batch_size,entity_type,seq_len,seq_len)
        因为一般都是对(batch_size,seq_len)进行pad，，，
    :param length: 这个是设置补充之后的长度，一般为None，根据batch的实际长度进行pad
    :param value:
    :param mode:
    :return:
    '''
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)  # length=np.array([max_batch_length])
    elif not hasattr(length, '__getitem__'):  # 如果这个length的类别不是列表....,就进行转变
        length = [length]
    # logger.info('这个batch下面的最长长度为{}'.format(length[0]))

    slices = [np.s_[:length[i]] for i in
              range(seq_dims)]  # 获得针对针对不同维度的slice，对于seq_dims=0,slice=[None:max_len:None],max_len是seq_dims的最大值
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]  # 有多少个维数，就需要多少个(0,0),一般是一个

    outputs = []
    for x in inputs:
        # X为一个列表
        # 这里就是截取长度
        x = x[slices]
        for i in range(seq_dims):  # 对不同的维度逐步进行扩充
            if mode == 'post':
                # np.shape(x)[i]是获得当前的实际长度
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)

def make_graph_for_spacy(tokenizer,pad_length,batch_input_ids,nlp):
    all_edge_index = []
    all_dep_matrix = []
    for inputs in batch_input_ids:
        tokens_with_padding = tokenizer.convert_ids_to_tokens(inputs)
        raw_sentence = ""
        for (i,token) in enumerate(tokens_with_padding):
            if token in ['[CLS]','[PAD]','[SEP]','[UNK]']: 
                pass
            elif token in ['[s1]','[s2]']:
                raw_sentence += " "
                raw_sentence += '@'
                tokens_with_padding[i] = '@'
            elif token in ['[e1]','[e2]']:
                raw_sentence += " "
                raw_sentence += '$'
                tokens_with_padding[i] = '$'
            elif "##" in token:
                raw_sentence += token[2:]
            else:
                raw_sentence += " "
                raw_sentence += token
        raw_sentence = raw_sentence[1:]
        edge_index = make_graph(raw_sentence,tokens_with_padding,pad_length,nlp)
        all_edge_index.append(edge_index)
    return all_edge_index,all_dep_matrix

def make_graph(sentence,tokens_with_padding,pad_length,nlp):
    adjacent_matrix_with_self = [[0 for i in range(pad_length)] for j in range(pad_length)]
    entity_start_index = [i for i, x in enumerate(tokens_with_padding) if x == "@"]
    entity_end_index = [i for i, x in enumerate(tokens_with_padding) if x == "$"]
    sentence_dependency_parse = nlp(sentence)
    dependency = []
    dep_text = []
    dep_matrix = []
    for token in sentence_dependency_parse:
        dep_text.append(token.text)
        dependency.append((token.i, token.head.i, token.dep_))
    bool_list = [0] * pad_length
    
    flag =  False
    sent_end_site = len(tokens_with_padding)
    for i,item in enumerate(tokens_with_padding):
        if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]" and item != "[UNK]":
            bool_list[i] = 1
        if item == '[SEP]' and flag ==False:
            sent_end_site = i
            flag = True
    # 所有单词的开始位置
    word_index = [i for i,x in enumerate(bool_list) if x==1]
    #需要匹配两边位置一致
    if len(word_index) != len(dep_text):
        new_word_index = []
        length = min(len(word_index),len(dep_text))

        i = 0
        j = 0
        last_ent = ""
        last_token = ""
        while i < length:
            new_word_index.append(word_index[i])
            start_site = word_index[i]
            end_site = word_index[i+1] if i<length-1 else sent_end_site
            ent = tokens_with_padding[start_site]
            for k in range(start_site+1,end_site):
                ent += tokens_with_padding[k][2:]
            if last_token + ent != last_ent + dep_text[j]:
                if len(last_token + ent) > len(last_ent + dep_text[j]):
                    last_ent+=dep_text[j]
                    j+=1
                else:
                    last_token += ent
                    i+=1
            else:
                last_ent = ""
                last_token = ""
                i+=1
                j+=1
        word_index = new_word_index
    # generating adjacent matrix

    # if len(entity_start_index) == 2 and len(entity_end_index) == 2:
    #     #找出实体位置
    #     neighbor_CLS_index = [x for x in word_index if (int(x) > int(entity_start_index[0]) and int(x) < int(entity_end_index[0])) 
    #                             or (int(x) > int(entity_start_index[1]) and int(x) < int(entity_end_index[1]))]
    #     #和cls标记相关？
    #     for neighbor in neighbor_CLS_index:
    #         adjacent_matrix_with_self[0][int(neighbor)] = 1
    #         adjacent_matrix_with_self[int(neighbor)][0] = 1

    else:
        print(len(entity_start_index), len(entity_end_index))
    # set_trace()
    for i in range(pad_length):
        if tokens_with_padding[i] == "[PAD]":
            pass
        else:
            adjacent_matrix_with_self[i][i] = 1.0
    for i, item in enumerate(tokens_with_padding):
        if item[0:2] == "##":
            adjacent_matrix_with_self[i][i-1] = 1.0
            adjacent_matrix_with_self[i-1][i] = 1.0
        else:
            pass
    for tail, head, rel in dependency:
        rel = rel.lower()
        try:
            adjacent_matrix_with_self[word_index[tail]][word_index[head]] = 1.0
            adjacent_matrix_with_self[word_index[head]][word_index[tail]] = 1.0
        except:
            set_trace()
            print(tokens_with_padding)
    return adjacent_matrix_with_self,dep_matrix

def make_gpnn_graph_for_spacy(tokenizer,pad_length,batch_input_ids,nlp,num_chosn_neighbors,device):
    all_edge_index = ()
    all_neighbor_index = ()
    all_mask_length = ()
    all_mask = ()
    for inputs in batch_input_ids:
        tokens_with_padding = tokenizer.convert_ids_to_tokens(inputs)
        raw_sentence = ""
        for (i,token) in enumerate(tokens_with_padding):
            if token in ['[CLS]','[PAD]','[SEP]','[UNK]']: 
                pass
            elif token in ['[s1]','[s2]']:
                raw_sentence += " "
                raw_sentence += '@'
                tokens_with_padding[i] = '@'
            elif token in ['[e1]','[e2]']:
                raw_sentence += " "
                raw_sentence += '$'
                tokens_with_padding[i] = '$'
            elif "##" in token:
                raw_sentence += token[2:]
            else:
                raw_sentence += " "
                raw_sentence += token
        raw_sentence = raw_sentence[1:]
        edge_index, neighbor_index, mask_length, mask= make_gpnn_graph(raw_sentence,tokens_with_padding,pad_length,nlp,num_chosn_neighbors=num_chosn_neighbors)
        all_edge_index = all_edge_index + (edge_index,)
        all_neighbor_index = all_neighbor_index + (neighbor_index,)
        all_mask_length = all_mask_length + (mask_length,)
        all_mask = all_mask + (mask,)
    # all_inputs.append(inputs)
    all_edge_index = torch.stack(all_edge_index).to(device)
    all_neighbor_index =torch.stack(all_neighbor_index).to(device)
    all_mask_length =torch.stack(all_mask_length).to(device)
    all_mask = torch.stack(all_mask).to(device)
    return (all_edge_index,all_neighbor_index,all_mask_length,all_mask)

def make_gpnn_graph(sentence, tokens_with_padding, max_length, nlp,num_chosn_neighbors=8, secondary_hoop=True):
    adjacent_matrix_with_self = torch.zeros((max_length, max_length)).long()
    entity_start_index = [i for i, x in enumerate(tokens_with_padding) if x == "@"]
    entity_end_index = [i for i, x in enumerate(tokens_with_padding) if x == "$"]
    adjacent_matrix_without_self = torch.zeros((max_length, max_length)).long()
    
    # sentence parsing
    sentence_dependency_parse = nlp(sentence)
    pos = []
    tag = []
    dep_text = []
    dependency = []
    parse_tokens = []
    for token in sentence_dependency_parse:
        dep_text.append(token.text)
        dependency.append((token.i, token.head.i, token.dep_))
    
    
    bool_list = [0] * max_length
    flag =  False
    sent_end_site = len(tokens_with_padding)
    for i,item in enumerate(tokens_with_padding):
        if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]" and item != "[UNK]":
            bool_list[i] = 1
        if item == '[SEP]' and flag ==False:
            sent_end_site = i
            flag = True
    # 所有单词的开始位置
    word_index = [i for i,x in enumerate(bool_list) if x==1]
    #需要匹配两边位置一致
    new_word_index = []
    if len(word_index) != len(dep_text):
        length = min(len(word_index),len(dep_text))
        i = 0
        j = 0
        last_token = ""
        last_ent = ""
        # set_trace()
        while i < length :
            new_word_index.append(word_index[i])
            start_site = word_index[i]
            end_site = word_index[i+1] if i<length-1 else sent_end_site
            ent = tokens_with_padding[start_site]
            for k in range(start_site+1,end_site):
                ent += tokens_with_padding[k][2:]
            # set_trace()
            # print(i)
            # print(j)
            # print(last_token + ent)
            # print(last_ent + dep_text[j])
            if last_token + ent != last_ent + dep_text[j]:
                if len(last_token + ent) > len(last_ent + dep_text[j]):
                    last_ent+=dep_text[j]
                    j+=1
                else:
                    last_token += ent
                    i+=1
                # set_trace()

            else:
                last_token = ""
                last_ent = ""
                i+=1
                j+=1
        # set_trace()
        word_index = new_word_index

    # generating adjacent matrix
    if len(entity_start_index) == 2 and len(entity_end_index) == 2:
        #找出实体位置
        neighbor_CLS_index = [x for x in word_index if (int(x) > int(entity_start_index[0]) and int(x) < int(entity_end_index[0])) 
                                or (int(x) > int(entity_start_index[1]) and int(x) < int(entity_end_index[1]))]
        #和cls标记相关？
        for neighbor in neighbor_CLS_index:
            adjacent_matrix_with_self[0, int(neighbor)] = 1
            adjacent_matrix_with_self[int(neighbor), 0] = 1
            adjacent_matrix_without_self[0, int(neighbor)] = 1
            adjacent_matrix_without_self[int(neighbor), 0] = 1

    else:
        print(len(entity_start_index), len(entity_end_index))
    # set_trace()
    length = max(len(word_index),len(dep_text))
    for i in range(max_length):
        if tokens_with_padding[i] == "[PAD]":
            pass
        else:
            adjacent_matrix_with_self[i][i] = 1.0
    for i, item in enumerate(tokens_with_padding):
        if item[0:2] == "##":
            adjacent_matrix_with_self[i][i-1] = 1.0
            adjacent_matrix_with_self[i-1][i] = 1.0
            adjacent_matrix_without_self[i][i-1] = 1.0
            adjacent_matrix_without_self[i-1][i] = 1.0
        else:
            pass
    for tail, head, rel in dependency:
        rel = rel.lower()
        try:
            adjacent_matrix_with_self[word_index[tail]][word_index[head]] = 1.0
            adjacent_matrix_with_self[word_index[head]][word_index[tail]] = 1.0
            adjacent_matrix_without_self[word_index[tail]][word_index[head]] = 1.0
            adjacent_matrix_without_self[word_index[head]][word_index[tail]] = 1.0
        except:
            print(parse_tokens, word_index)
            print(tokens_with_padding)
        # edge_index = dense_to_sparse(adjacent_matrix_with_self)[0]
    adj_1_hoop = adjacent_matrix_without_self
    adj_2_hoop = ((adjacent_matrix_with_self @ adjacent_matrix_with_self)>0).int()-adjacent_matrix_with_self
    neighbor_index = torch.zeros(max_length, num_chosn_neighbors, dtype=int)
    mask_length = torch.zeros(max_length, dtype=int)
    mask = torch.zeros(max_length, num_chosn_neighbors)
    num_chosn_neighbors = num_chosn_neighbors - 1 # center node and k-1 neighbors
    for i in range(max_length):
        if tokens_with_padding[i] == "[PAD]":
            pass
        index1 = torch.nonzero(adj_1_hoop[i]).long()
        num_1_hoop_neighbors = index1.size()[0]
        num_2_hoop_neighbors = num_chosn_neighbors - num_1_hoop_neighbors
        neighbor_index[i,0]=i
        mask_length[i] = min(num_1_hoop_neighbors+1,num_chosn_neighbors+1)
        # 1 hop
        for j1 in range(num_1_hoop_neighbors):
            if j1>=num_chosn_neighbors: break
            neighbor_index[i,j1+1]=index1[j1]
        # 2 hop
        if secondary_hoop and num_2_hoop_neighbors>0:
            index2 = torch.nonzero(adj_2_hoop[i]).long()
            mask_length[i] = min(num_1_hoop_neighbors+index2.size()[0]+1,num_chosn_neighbors+1)
            for j2 in range(index2.size()[0]):
                if j2>=num_2_hoop_neighbors: break
                neighbor_index[i,j2+num_1_hoop_neighbors+1]=index2[j2]
        mask[i] = torch.tensor([1]*mask_length[i]+[0]*(num_chosn_neighbors+1-mask_length[i])).long()
            # if i<10:
            #     print(f"Node:{i}   neighbor_index:{neighbor_index[i]}  mask_length:{mask_length[i]}")
                
    mask = mask.unsqueeze(2).repeat(1,1,768)
        
    return adjacent_matrix_with_self, neighbor_index, mask_length, mask

# 针对于将两者结合的方法
def make_graph_for_neg(tokenizer,pad_length,batch_input_ids,nlp,dep_dict,device):
    all_edge_index = ()
    all_dep_matrix = []
    for inputs in batch_input_ids:
        tokens_with_padding = tokenizer.convert_ids_to_tokens(inputs)
        raw_sentence = ""
        for (i,token) in enumerate(tokens_with_padding):
            if token in ['[CLS]','[PAD]','[SEP]','[UNK]']: 
                pass
            elif token in ['[s1]','[s2]']:
                raw_sentence += " "
                raw_sentence += '@'
                tokens_with_padding[i] = '@'
            elif token in ['[e1]','[e2]']:
                raw_sentence += " "
                raw_sentence += '$'
                tokens_with_padding[i] = '$'
            elif "##" in token:
                raw_sentence += token[2:]
            else:
                raw_sentence += " "
                raw_sentence += token
        raw_sentence = raw_sentence[1:]
        edge_index,dep_matrix= make_neg_graph(raw_sentence,tokens_with_padding,pad_length,nlp,dep_dict)
        all_edge_index = all_edge_index + (edge_index,)
        all_dep_matrix.append(dep_matrix)
    
    # all_inputs.append(inputs)
    all_edge_index = torch.stack(all_edge_index).to(device)

    all_dep_matrix =torch.tensor(sequence_padding(all_dep_matrix, length=pad_length), device=device).long()

    return (all_edge_index,all_dep_matrix)

def make_neg_graph(sentence, tokens_with_padding, max_length, nlp,dep_dict):
    adjacent_matrix_with_self = torch.zeros((max_length, max_length)).long()
    dep_matrix = []
    # sentence parsing
    sentence_dependency_parse = nlp(sentence)
    pos = []
    tag = []
    dep_text = []
    dependency = []
    parse_tokens = []
    for token in sentence_dependency_parse:
        dep_text.append(token.text)
        dependency.append((token.i, token.head.i, token.dep_))
    
    
    bool_list = [0] * max_length
    flag =  False
    sent_end_site = len(tokens_with_padding)
    for i,item in enumerate(tokens_with_padding):
        if item[0:2] != "##" and item != "[CLS]" and item != "[SEP]" and item != "[PAD]" and item != "[UNK]":
            bool_list[i] = 1
        if item == '[SEP]' and flag ==False:
            sent_end_site = i
            flag = True
    # 所有单词的开始位置
    word_index = [i for i,x in enumerate(bool_list) if x==1]
    #需要匹配两边位置一致
    new_word_index = []
    if len(word_index) != len(dep_text):
        length = min(len(word_index),len(dep_text))
        i = 0
        j = 0
        last_token = ""
        last_ent = ""
        # set_trace()
        while i < length :
            new_word_index.append(word_index[i])
            start_site = word_index[i]
            end_site = word_index[i+1] if i<length-1 else sent_end_site
            ent = tokens_with_padding[start_site]
            for k in range(start_site+1,end_site):
                ent += tokens_with_padding[k][2:]
            # set_trace()
            # print(i)
            # print(j)
            # print(last_token + ent)
            # print(last_ent + dep_text[j])
            if last_token + ent != last_ent + dep_text[j]:
                if len(last_token + ent) > len(last_ent + dep_text[j]):
                    last_ent+=dep_text[j]
                    j+=1
                else:
                    last_token += ent
                    i+=1
                # set_trace()

            else:
                last_token = ""
                last_ent = ""
                i+=1
                j+=1
        # set_trace()
        word_index = new_word_index
    # 制作dep_matrix
    for id,num in enumerate(word_index):
        end_site = word_index[id+1] if id+1<len(word_index) else sent_end_site-1
        repeat_num = end_site - num +1
        for i in range(repeat_num):
            dep_matrix.append(dep_dict[dependency[id][2]])

    # generating adjacent matrix
    # if len(entity_start_index) == 2 and len(entity_end_index) == 2:
    #     #找出实体位置
    #     neighbor_CLS_index = [x for x in word_index if (int(x) > int(entity_start_index[0]) and int(x) < int(entity_end_index[0])) 
    #                             or (int(x) > int(entity_start_index[1]) and int(x) < int(entity_end_index[1]))]
    #     #和cls标记相关？
    #     for neighbor in neighbor_CLS_index:
    #         adjacent_matrix_with_self[0, int(neighbor)] = 1
    #         adjacent_matrix_with_self[int(neighbor), 0] = 1
    # else:
    #     print(len(entity_start_index), len(entity_end_index))
    # set_trace()
    length = max(len(word_index),len(dep_text))

    for i in range(max_length):
        if tokens_with_padding[i] == "[PAD]":
            pass
        else:
            adjacent_matrix_with_self[i][i] = 1.0
    for i, item in enumerate(tokens_with_padding):
        if item[0:2] == "##":
            adjacent_matrix_with_self[i][i-1] = 1.0
            adjacent_matrix_with_self[i-1][i] = 1.0
        else:
            pass
    for tail, head, rel in dependency:
        rel = rel.lower()
        try:
            adjacent_matrix_with_self[word_index[tail]][word_index[head]] = 1.0
            adjacent_matrix_with_self[word_index[head]][word_index[tail]] = 1.0
        except:
            print(parse_tokens, word_index)
            print(tokens_with_padding)
        # edge_index = dense_to_sparse(adjacent_matrix_with_self)[0]

        
    return adjacent_matrix_with_self,dep_matrix

if __name__ == '__main__':
    sentences_file = './general_domain_dataset/semeval2008/mid_dataset/train/sentences.txt'
    labels_file = './general_domain_dataset/semeval2008/mid_dataset/train/labels.txt'
    sents, labels = read_data(sentences_file, labels_file)
    label2id, id2label = get_label2id('./general_domain_dataset/semeval2008/mid_dataset/labels.txt')
