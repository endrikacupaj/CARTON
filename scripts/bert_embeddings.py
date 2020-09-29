import re
import time
import json
import torch
import flair
from glob import glob
from dataset import CSQADataset
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings

# import constants
from constants import *

train, val, test = [], [], []
# read data
train_files = glob(str(ROOT_PATH.parent) + '/data/final/csqa' + '/train/*' + '/*.json')
for f in train_files:
    with open(f) as json_file:
        train.append(json.load(json_file))

val_files = glob(str(ROOT_PATH.parent) + '/data/final/csqa' + '/val/*' + '/*.json')
for f in val_files:
    with open(f) as json_file:
        val.append(json.load(json_file))

test_files = glob(str(ROOT_PATH.parent) + '/data/final/csqa' + '/test/*' + '/*.json')
for f in test_files:
    with open(f) as json_file:
        test.append(json.load(json_file))

all_entities = set()
tic = time.perf_counter()
for partition in [train, val, test]:
    for j, conversation in enumerate(partition):
        turns = len(conversation) // 2
        for i in range(turns):
            user = conversation[2*i]
            system = conversation[2*i + 1]

            user_entities = []
            if 'entities' in user: all_entities.update(user['entities'])
            if 'entities_in_utterance' in user: all_entities.update(user['entities_in_utterance'])

            all_entities.update(system['entities_in_utterance'])

        if (j+1) % 1000 == 0:
            toc = time.perf_counter()
            print(f'==> Finished {((j+1)/len(partition))*100:.2f}% -- {toc - tic:0.2f}s')

# set device
torch.cuda.set_device(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flair.device = DEVICE

# load bert model
bert = DocumentPoolEmbeddings([BertEmbeddings('bert-base-uncased', layers='-1')])

# read entities
id_entity = json.loads(open(f'{ROOT_PATH}/knowledge_graph/items_wikidata_n.json').read())

# create embeddings for NA and PAD values
na = Sentence(NA_TOKEN)
pad = Sentence(PAD_TOKEN)
bert.embed(na)
bert.embed(pad)


entity_embeddings = {
    NA_TOKEN: na.embedding.detach().cpu().tolist(),
    PAD_TOKEN: pad.embedding.detach().cpu().tolist()
}

tic = time.perf_counter()
for i, id in enumerate(list(all_entities)):
    if id in entity_embeddings: continue
    flair_sentence = Sentence(id_entity[id])
    bert.embed(flair_sentence)
    entity_embeddings[id] = flair_sentence.embedding.detach().cpu().tolist()

    if (i+1) % 1000 == 0:
        toc = time.perf_counter()
        print(f'==> Finished {((i+1)/len(all_entities))*100:.2f}% -- {toc - tic:0.2f}s')

with open(f'{ROOT_PATH}/knowledge_graph/entity_embeddings.json', 'w') as outfile:
    json.dump(entity_embeddings, outfile, indent=4)