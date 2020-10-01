from __future__ import division
import os
import re
import time
import json
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
from args import get_parser
from unidecode import unidecode
from collections import OrderedDict
from transformers import BertTokenizer
from elasticsearch import Elasticsearch

# import constants
from constants import *

# set logger
logging.getLogger('elasticsearch').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size=args.emb_dim, factor=args.factor, warmup=args.warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Predictor(object):
    """Predictor class"""
    def __init__(self, model, vocabs):
        self.model = model
        self.vocabs = vocabs

    def predict(self, input, ent_cand, helper):
        """Perform prediction on given input example"""
        self.model.eval()
        model_out = {}

        # prepare input
        tokenized_sentence = [START_TOKEN] + [t.lower() for t in input] + [CTX_TOKEN]
        numericalized = [self.vocabs[INPUT].stoi[token] if token in self.vocabs[INPUT].stoi else self.vocabs[INPUT].stoi[UNK_TOKEN] for token in tokenized_sentence]
        src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(DEVICE)

        # prepare entity candidates
        numericalized_ent_cand = [self.vocabs[ENTITY_POINTER].stoi[entity] for entity in ent_cand]
        ent_cand_tensor = torch.LongTensor(numericalized_ent_cand).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # get ner, coref predictions
            encoder_out = self.model.encoder(src_tensor)
            encoder_ctx = encoder_out[:, -1:, :]

            # get logical form, predicate and type prediction
            lf_out = [self.vocabs[LOGICAL_FORM].stoi[START_TOKEN]]
            pd_out = [self.vocabs[PREDICATE_POINTER].stoi[NA_TOKEN]]
            tp_out = [self.vocabs[TYPE_POINTER].stoi[NA_TOKEN]]
            en_out = [self.vocabs[ENTITY_POINTER].stoi[NA_TOKEN]]

            for _ in range(self.model.decoder.max_positions):
                lf_tensor = torch.LongTensor(lf_out).unsqueeze(0).to(DEVICE)

                # decoder_step = self.model._predict_decoder(src_tensor, lf_tensor, encoder_step[ENCODER_OUT])
                decoder_out, decoder_h = self.model.decoder(src_tensor, lf_tensor, encoder_out)
                stacked_pointer_out = self.model.stptr_net(encoder_ctx, decoder_h, ent_cand_tensor)

                pred_lf = decoder_out.argmax(1)[-1].item()
                pred_pd = stacked_pointer_out[PREDICATE_POINTER].argmax(1)[-1].item()
                pred_tp = stacked_pointer_out[TYPE_POINTER].argmax(1)[-1].item()
                pred_en = stacked_pointer_out[ENTITY_POINTER].argmax(1)[-1].item()

                if pred_lf == self.vocabs[LOGICAL_FORM].stoi[END_TOKEN]:
                    break

                lf_out.append(pred_lf)
                pd_out.append(pred_pd)
                tp_out.append(pred_tp)
                en_out.append(pred_en)

        # translate top predictions into vocab tokens
        model_out[LOGICAL_FORM] = [self.vocabs[LOGICAL_FORM].itos[i] for i in lf_out][1:]
        model_out[PREDICATE_POINTER] = [self.vocabs[PREDICATE_POINTER].itos[i] for i in pd_out][1:]
        model_out[TYPE_POINTER] = [self.vocabs[TYPE_POINTER].itos[i] for i in tp_out][1:]
        model_out[ENTITY_POINTER] = [self.vocabs[ENTITY_POINTER].itos[i] for i in en_out][1:]

        return model_out

class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.accuracy = 0

    def update(self, gold, result):
        if gold == result:
            self.correct += 1
        else:
            self.wrong += 1

        self.accuracy = self.correct / (self.correct + self.wrong)

class Scorer(object):
    """Scorer class"""
    def __init__(self):
        self.tasks = [TOTAL, LOGICAL_FORM, PREDICATE_POINTER, TYPE_POINTER, ENTITY_POINTER]
        self.results = {
            OVERALL: {task:AccuracyMeter() for task in self.tasks},
            CLARIFICATION: {task:AccuracyMeter() for task in self.tasks},
            COMPARATIVE: {task:AccuracyMeter() for task in self.tasks},
            LOGICAL: {task:AccuracyMeter() for task in self.tasks},
            QUANTITATIVE: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_COREFERENCED: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_DIRECT: {task:AccuracyMeter() for task in self.tasks},
            SIMPLE_ELLIPSIS: {task:AccuracyMeter() for task in self.tasks},
            # -------------------------------------------
            VERIFICATION: {task:AccuracyMeter() for task in self.tasks},
            QUANTITATIVE_COUNT: {task:AccuracyMeter() for task in self.tasks},
            COMPARATIVE_COUNT: {task:AccuracyMeter() for task in self.tasks},
        }
        self.data_dict = []

    def data_score(self, data, helper, predictor):
        """Score complete list of data"""
        for i, (example, q_type)  in enumerate(zip(data, helper['question_type'])):
            # prepare references
            ref_lf = [t.lower() for t in example.logical_form]
            ref_pd = example.predicate_pointer
            ref_tp = example.type_pointer
            ref_en = helper[ENTITY][LABEL][example.id[0]]

            # get model hypothesis
            hypothesis = predictor.predict(example.input, example.entity_pointer, helper)

            # check correctness
            correct_lf = 1 if ref_lf == hypothesis[LOGICAL_FORM] else 0
            correct_pd = 1 if ref_pd == hypothesis[PREDICATE_POINTER] else 0
            correct_tp = 1 if ref_tp == hypothesis[TYPE_POINTER] else 0
            correct_en = 1 if ref_en == hypothesis[ENTITY_POINTER] else 0

            # save results
            gold = 1
            res = 1 if correct_lf and correct_pd and correct_tp and correct_en else 0
            # Question type
            self.results[q_type][TOTAL].update(gold, res)
            self.results[q_type][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results[q_type][PREDICATE_POINTER].update(ref_pd, hypothesis[PREDICATE_POINTER])
            self.results[q_type][TYPE_POINTER].update(ref_tp, hypothesis[TYPE_POINTER])
            self.results[q_type][ENTITY_POINTER].update(ref_en, hypothesis[ENTITY_POINTER])
            # Overall
            self.results[OVERALL][TOTAL].update(gold, res)
            self.results[OVERALL][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results[OVERALL][PREDICATE_POINTER].update(ref_pd, hypothesis[PREDICATE_POINTER])
            self.results[OVERALL][TYPE_POINTER].update(ref_tp, hypothesis[TYPE_POINTER])
            self.results[OVERALL][ENTITY_POINTER].update(ref_en, hypothesis[ENTITY_POINTER])

            # save data
            self.data_dict.append({
                INPUT: example.input,
                LOGICAL_FORM: hypothesis[LOGICAL_FORM],
                f'{LOGICAL_FORM}_gold': ref_lf,
                PREDICATE_POINTER: hypothesis[PREDICATE_POINTER],
                f'{PREDICATE_POINTER}_gold': ref_pd,
                TYPE_POINTER: hypothesis[TYPE_POINTER],
                f'{TYPE_POINTER}_gold': ref_tp,
                ENTITY_POINTER: hypothesis[ENTITY_POINTER],
                f'{TYPE_POINTER}_gold': ref_en,
                # ------------------------------------
                f'{LOGICAL_FORM}_correct': correct_lf,
                f'{PREDICATE_POINTER}_correct': correct_pd,
                f'{TYPE_POINTER}_correct': correct_tp,
                f'{ENTITY_POINTER}_correct': correct_en,
                IS_CORRECT: res,
                QUESTION_TYPE: q_type
            })

            if (i+1) % 500 == 0:
                logger.info(f'* {OVERALL} Data Results {i+1}:')
                for task, task_result in self.results[OVERALL].items():
                    logger.info(f'\t\t{task}: {task_result.accuracy:.4f}')

    def write_results(self):
        save_dict = json.dumps(self.data_dict, indent=4)
        save_dict_no_space_1 = re.sub(r'": \[\s+', '": [', save_dict)
        save_dict_no_space_2 = re.sub(r'",\s+', '", ', save_dict_no_space_1)
        save_dict_no_space_3 = re.sub(r'"\s+\]', '"]', save_dict_no_space_2)
        with open(f'{ROOT_PATH}/{args.path_error_analysis}/error_analysis.json', 'w', encoding='utf-8') as json_file:
            json_file.write(save_dict_no_space_3)

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.instances = 0

class Inference(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED)
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}]) # connect to elastic search server
        self.inference_actions = []

    def construct_actions(self, inference_data, predictor):
        tic = time.perf_counter()
        # based on model outpus create a final logical form to execute
        question_type_inference_data = [data for data in inference_data if args.question_type in data[QUESTION_TYPE]]
        for i, sample in enumerate(question_type_inference_data):
            predictions = predictor.predict(sample['context_question'])
            actions = []
            logical_form_prediction = predictions[LOGICAL_FORM]
            ent_count_pos = 0
            for j, action in enumerate(logical_form_prediction):
                if action not in [ENTITY, RELATION, TYPE, VALUE, PREV_ANSWER]:
                    actions.append([ACTION, action])
                elif action == ENTITY:
                    actions.append([ENTITY, ENTITY])
                elif action == RELATION:
                    actions.append([RELATION, RELATION])
                elif action == TYPE:
                    actions.append([TYPE, TYPE])
                elif action == VALUE:
                    try:
                        actions.append([VALUE, self.get_value(sample[QUESTION])])
                    except Exception as ex:
                        print(ex)
                        actions.append([VALUE, '0'])
                elif action == PREV_ANSWER:
                    actions.append([ENTITY, PREV_ANSWER])

            self.inference_actions.append({
                QUESTION_TYPE: sample[QUESTION_TYPE],
                QUESTION: sample[QUESTION],
                ANSWER: sample[ANSWER],
                ACTIONS: actions,
                RESULTS: sample[RESULTS],
                PREV_RESULTS: sample[PREV_RESULTS],
                GOLD_ACTIONS: sample[GOLD_ACTIONS] if GOLD_ACTIONS in sample else [],
                IS_CORRECT: 1 if GOLD_ACTIONS in sample and sample[GOLD_ACTIONS] == actions else 0
            })

            if (i+1) % 100 == 0:
                toc = time.perf_counter()
                print(f'==> Finished action construction {((i+1)/len(question_type_inference_data))*100:.2f}% -- {toc - tic:0.2f}s')

        self.write_inference_actions()

    def create_ner_idx_ent_dict(self, ner_indices, context_question):
        ent_idx = []
        ner_idx_ent = OrderedDict()
        for index, span_type in ner_indices.items():
            if not ent_idx or index-1 == ent_idx[-1][0]:
                ent_idx.append([index, span_type]) # check wether token start with ## then include previous token also from context_question
            else:
                # get ent tokens from input context
                ent_tokens = [context_question[idx] for idx, _ in ent_idx]
                # get string from tokens using tokenizer
                ent_string = self.tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')
                # get elastic search results
                es_results = self.elasticsearch_query(ent_string, ent_idx[0][1]) # use type from B tag only
                # add idices to dict
                if es_results:
                    for idx, _ in ent_idx:
                        ner_idx_ent[idx] = es_results
                # clean ent_idx
                ent_idx = [[index, span_type]]
        if ent_idx:
            # get ent tokens from input context
            ent_tokens = [context_question[idx] for idx, _ in ent_idx]
            # get string from tokens using tokenizer
            ent_string = self.tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')
            # get elastic search results
            es_results = self.elasticsearch_query(ent_string, ent_idx[0][1])
            # add idices to dict
            if es_results:
                for idx, _ in ent_idx:
                    ner_idx_ent[idx] = es_results
        return ner_idx_ent

    def elasticsearch_query(self, query, filter_type, res_size=50):
        res = self.es.search(index='csqa_wikidata', doc_type='entities', body={'size': res_size, 'query': {'match': {'label': {'query': unidecode(query), 'fuzziness': 'AUTO'}}}})
        results = []
        for hit in res['hits']['hits']: results.append([hit['_source']['id'], hit['_source']['type']])
        filtered_results = [res for res in results if filter_type in res[1]]
        return [res[0] for res in filtered_results] if filtered_results else [res[0] for res in results]

    def get_value(self, question):
        if 'min' in question.split():
            value = '0'
        elif 'max' in question.split():
            value = '0'
        elif 'exactly' in question.split():
            value = re.search(r'\d+', question.split('exactly')[1]).group()
        elif 'approximately' in question.split():
            value = re.search(r'\d+', question.split('approximately')[1]).group()
        elif 'around' in question.split():
            value = re.search(r'\d+', question.split('around')[1]).group()
        elif 'atmost' in question.split():
            value = re.search(r'\d+', question.split('atmost')[1]).group()
        elif 'atleast' in question.split():
            value = re.search(r'\d+', question.split('atleast')[1]).group()
        else:
            print(f'Could not extract value from question: {question}')
            value = '0'

        return value

    def write_inference_actions(self):
        with open(f'{ROOT_PATH}/{args.path_inference}/{args.model_path.rsplit("/", 1)[-1].rsplit(".", 2)[0]}_{args.inference_partition}_{args.question_type}.json', 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(self.inference_actions, indent=4))

def save_checkpoint(state):
    filename = f'{ROOT_PATH}/{args.snapshots}/{MODEL_NAME}_e{state[EPOCH]}_v{state[CURR_VAL]:.4f}_{args.task}.pth.tar'
    torch.save(state, filename)

class SingleTaskLoss(nn.Module):
    '''Single Task Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.criterion(output, target)

class MultiTaskLoss(nn.Module):
    '''Multi Task Learning Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.lf_loss = SingleTaskLoss(ignore_index)
        self.pred_pointer = SingleTaskLoss(ignore_index)
        self.type_pointer = SingleTaskLoss(ignore_index)
        self.ent_pointer = SingleTaskLoss(ignore_index)

        self.mml_emp = torch.Tensor([True, True, True, True])
        self.log_vars = torch.nn.Parameter(torch.zeros(len(self.mml_emp)))

    def forward(self, output, target):
        # weighted loss
        task_losses = torch.stack((
            self.lf_loss(output[LOGICAL_FORM], target[LOGICAL_FORM]),
            self.pred_pointer(output[PREDICATE_POINTER], target[PREDICATE_POINTER]),
            self.type_pointer(output[TYPE_POINTER], target[TYPE_POINTER]),
            self.ent_pointer(output[ENTITY_POINTER], target[ENTITY_POINTER])
        ))

        dtype = task_losses.dtype
        stds = (torch.exp(self.log_vars)**(1/2)).to(DEVICE).to(dtype)
        weights = 1 / ((self.mml_emp.to(DEVICE).to(dtype)+1)*(stds**2))

        losses = weights * task_losses + torch.log(stds)

        return {
            LOGICAL_FORM: losses[0],
            PREDICATE_POINTER: losses[1],
            TYPE_POINTER: losses[2],
            ENTITY_POINTER: losses[3],
            MULTITASK: losses.mean()
        }[args.task]

def init_weights(model):
    # initialize model parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def construct_entity_target(id_batch, helper_data, vocabs, max_size):
    ent_t = []
    for idx in id_batch:
        id = vocabs[ID].itos[idx]
        e_t = helper_data[ENTITY][GOLD][id]
        while len(e_t) < max_size: e_t.append(vocabs[ENTITY_POINTER].stoi[PAD_TOKEN]) # add padding
        ent_t.append(torch.tensor(e_t))
    return torch.stack(ent_t).to(DEVICE)
