# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""

import os
import datetime
import copy

from ipdb import set_trace
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer,ElectraTokenizer

import torch.nn as nn
from re_dev import dev
from src.dataset_utils.data_process_utils import read_data, get_label2id
from src.utils.function_utils import set_seed, save_model, load_model_and_parallel, set_cv_config, count_parameters
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import build_optimizer_and_scheduler, relation_classification_decode, batch_to_device, \
    set_tokenize_special_tag, choose_model, choose_dataloader, build_optimizer, save_parameter_writer

def train(config=None, logger=None):
    # 这里初始化device，为了在Dataset时加载到device之中
    device = torch.device('cuda:{}'.format(config.gpu_id)) if config.use_gpu else torch.device('cpu')
    if config.bert_name == 'bioelectra':
        tokenizer = ElectraTokenizer.from_pretrained(config.bert_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

    set_tokenize_special_tag(config, tokenizer)

    label2id, id2label = get_label2id(config.relation_labels)
    config.num_labels = len(label2id)
    examples = read_data(config)
    
    if config.debug:
        examples = examples[:config.batch_size * 3]
    train_dataloader = choose_dataloader(config, examples, label2id, tokenizer, device)

    model = choose_model(config)
    #异常检测
    torch.autograd.set_detect_anomaly(True)
    # 当添加新的token之后，就要重新调整embedding_size...
    model.bert_model.resize_token_embeddings(len(tokenizer))
    if config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    if config.use_n_gpu and torch.cuda.device_count() > 1:
        model, device = load_model_and_parallel(model, '0,1', ckpt_path=None, load_type='many2one')
    else:
        model.to(device)
    t_total = config.num_epochs * len(train_dataloader)
    if config.use_scheduler:
        optimizer, scheduler = build_optimizer_and_scheduler(config, model, t_toal=t_total,l1_lambda=config.l1_lambda)
    else:
        optimizer = build_optimizer(config, model)

    if config.l1_lambda !=0:
        lambda_l1 =config.l1_lambda
        l1_loss=nn.L1Loss(reduction='mean')

    if config.use_metric_summary_writer:
        metric_writer = SummaryWriter(
            os.path.join(config.tensorboard_dir,
                         "metric_{} {}-{} {}-{}-{}".format(config.model_name, now.month, now.day,
                                                           now.hour, now.minute,
                                                                   now.second)))
    if config.use_parameter_summary_writer:
        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        parameter_writer = SummaryWriter(
            os.path.join(config.tensorboard_dir, "parameter_{} {}-{} {}-{}-{}".format(config.model_name, now.month,
                                                                                      now.day,
                                                                                      now.hour, now.minute,
                                                                                      now.second)))

    best_model = None
    global_step = 0
    best_p = best_r = best_f1 = 0.
    best_epoch = 0
    # 使用wandb来记录模型训练的时候各种参数....
    # wandb.watch(model, torch.nn.CrossEntropyLoss, log="all", log_freq=2)
    # requires_grad_nums, parameter_nums = count_parameters(model)
    # set_trace()
    for epoch in range(1, config.num_epochs + 1):
        batch_loss = 0.
        batch_train_f1 = 0.
        batch_train_p = 0.
        batch_train_r = 0.

        all_train_labels = []
        all_predicate_tokens = []

        model.train()
        for step, batch_data in tqdm(enumerate(train_dataloader),total=len(train_dataloader),desc="Dataset:{},{}_{}....".format(config.dataset_name,config.bert_name,config.model_name)):

            if config.data_format == 'single':
                if 'gcn' in config.model_name or 'gat' in config.model_name:
                    input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels,ent_pair = batch_data
                    loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,ent_pair)
                else:
                    input_ids, token_type_ids, attention_masks, e1_mask, e2_mask, labels = batch_data
                    loss, logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)
                predicate_token = relation_classification_decode(logits)

            elif config.data_format == 'neg_single':
                if 'gcn' in config.model_name or 'gat' in config.model_name: 
                    input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask,ent_pair= batch_data
                    loss, logits = model(input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask,ent_pair)
                else:
                    input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask= batch_data
                    loss, logits = model(input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask)
                predicate_token = relation_classification_decode(logits)
            elif config.data_format == 'all_neg_single':
                input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,neg_input_ids,neg_token_type_ids,neg_attention_masks, dep_mask= batch_data
                loss, logits = model(input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,neg_input_ids,neg_token_type_ids,neg_attention_masks, dep_mask)
                predicate_token = relation_classification_decode(logits)

            else:
                raise ValueError
            

            loss = loss.mean()
            if config.use_fp16:
                scaler.scale(loss).backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if config.use_scheduler:
                        scheduler.step()
                    if config.use_parameter_summary_writer:
                        save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if config.use_scheduler:
                        scheduler.step()
                    if config.use_parameter_summary_writer:
                        save_parameter_writer(parameter_writer, model, global_step)
                    optimizer.zero_grad()
            if config.train_verbose:
                learning_rate = optimizer.param_groups[0]['lr']
                labels = labels.cpu().numpy()

                all_labels = [id for _, id in label2id.items()]

                all_train_labels.extend(labels)

                all_predicate_tokens.extend(predicate_token)

                # set_trace()

                p_r_f1_s = precision_recall_fscore_support(labels, predicate_token, labels=all_labels,average=config.evaluate_mode)

                acc = accuracy_score(labels, predicate_token)

                tmp_train_p = p_r_f1_s[0]
                tmp_train_r = p_r_f1_s[1]
                tmp_train_f1 = p_r_f1_s[2]

                batch_train_f1 += tmp_train_f1
                batch_train_p += tmp_train_p
                batch_train_r += tmp_train_r
                batch_loss += loss.item()
                if global_step % config.print_step == 0:
                    show_log(logger, step, len(train_dataloader), t_total, epoch, global_step, loss, tmp_train_p,
                             tmp_train_r, tmp_train_f1, acc, config.evaluate_mode, type_='train')

            global_step += 1
        if config.train_verbose:
            count = len(train_dataloader)
            batch_loss = batch_loss / count
            train_p = batch_train_p / count
            train_r = batch_train_r / count
            train_f1 = batch_train_f1 / count

            show_log(logger, -1, len(train_dataloader), t_total, epoch, global_step, batch_loss, train_p, train_r, train_f1,
                     0.00, config.evaluate_mode, type_='train', scheme=1)

            reports = classification_report(all_train_labels, all_predicate_tokens, labels=all_labels,digits=4)
            logger.info("-------Training Set epoch:{} Report----------".format(epoch))
            logger.info(reports)

        dev_p, dev_r, dev_f1 = dev(model, config, tokenizer, label2id, device, epoch=epoch, global_step=global_step,
                                   logger=logger,type_='dev')
        # If YOUR DATA SET NEED TO TEST,PUT HERE
        if config.dataset_name in ['ChemProt']:
            test_p, test_r, test_f1 = dev(model, config, tokenizer, label2id, device, epoch=epoch, global_step=global_step,
                                       logger=logger,type_='test')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_p = dev_p
            best_r = dev_r
            best_epoch = epoch
            if config.save_model:
                best_model = copy.deepcopy(model)
        if config.save_model:
            logger.info("Saving Model----------")
            save_model(config, model, epoch=epoch, mode='other')

    logger.info('{}Task{}Model，in{}epoch，the best{}-f1:{:.5f},{}-p:{:.5f},{}-r:{:.5f}.This Model will save in {}'.format(config.dataset_name,
                                                                                                 config.model_name,
                                                                                                 best_epoch,
                                                                                                 config.evaluate_mode,
                                                                                                 best_f1,
                                                                                                 config.evaluate_mode,
                                                                                                 best_p,
                                                                                                 config.evaluate_mode,
                                                                                                 best_r,
                                                                                                 config.output_dir))

    if config.save_model:
        save_model(config, best_model, mode='best_model')
    if config.use_metric_summary_writer:
        metric_writer.close()
    if config.use_parameter_summary_writer:
        parameter_writer.close()
    logger.info('----------------Parameters for this model run------------------')
    print_hyperparameters(config, logger)
    # Optional


if __name__ == '__main__':
    config = get_bert_config()

    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff

    set_seed(config.seed)



    if config.use_scheduler:
        if config.freeze_bert:
            wandb_name = f'single_task_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_free{config.freeze_layer_nums}_scheduler{config.warmup_proportion}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
        else:
            wandb_name = f'single_task_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_no_freeze_scheduler{config.warmup_proportion}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'

    else:
        if config.freeze_bert:
            wandb_name = f'single_task_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_free{config.freeze_layer_nums}_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'
        else:
            wandb_name = f'single_task_{config.bert_name}_{config.model_name}_epochs{config.num_epochs}_no_freeze_bs{config.batch_size}_schema{config.scheme}_lr{config.bert_lr}'

    config.output_dir = './outputs/save_models/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,config.model_name, config.dataset_name)
    config.logs_dir = './outputs/logs/{}/{}/{}/{}/'.format(str(datetime.date.today()), wandb_name,config.model_name, config.dataset_name)
    logger = get_logger(config)
    if config.run_type == 'normal':
        logger.info('----------------Parameters for this model run--------------------')
        print_hyperparameters(config, logger)
        train(config, logger)
    else:
        raise ValueError
