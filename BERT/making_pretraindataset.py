import json
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import os
import torch.nn.functional as F
import sentencepiece as spm



def create_pretrain_mask(tokens, mask_cnt, vocab_list):
  cand_idx = []
  for (i, token) in enumerate(tokens):
    if token == '[CLS]' or token == '[SEP]':
      continue
    if 0 < len(cand_idx) and not token.startswith(u"\u2581"): #해당 토큰이 문장의 시작이 아니면서 이전의 토큰과 연결된 subword라면
      cand_idx[-1].append(i)
    else :
      cand_idx.append([i])
  random.shuffle(cand_idx)

  mask_lms = []
  for index_set in cand_idx :
    if len(mask_lms) >= mask_cnt :
      break
    if len(mask_lms) + len(index_set) > mask_cnt :
      continue
    for index in index_set :
      masked_token = None
      if random.random() < 0.8:
        masked_token = "[MASK]"
      else :
        if random.random() < 0.5 :
          masked_token = tokens[index]
        else :
          maksed_token = random.choice(vocab_list)
      mask_lms.append({"index" : index, "label" : tokens[index]}) #바꾸기 전에 정답기록
      tokens[index] = masked_token
  mask_lms = sorted(mask_lms, key = lambda x : x["index"])
  mask_idx = [p["index"] for p in mask_lms]
  mask_label = [p["label"] for p in mask_lms]

  return tokens, mask_idx, mask_label    


def trim_tokens(tokens_a, tokens_b, max_seq):
  while True :
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_seq :
      break

    if len(tokens_a) > len(tokens_b):
      del tokens_a[0]
    else :
      tokens_b.pop()


def create_pretrain_instances(docs, doc_idx, doc, n_seq, mask_prob, vocab_list):
  max_seq = n_seq - 3 #for [CLS], [SEP], [SEP]
  tgt_seq = max_seq

  instances = []
  current_chunk = []
  current_length = 0
  for i in range(len(doc)):
    current_chunk.append(doc[i])
    current_length += len(doc[i])
    if i == len(doc) - 1 or current_length >= tgt_seq :   # doc의 token을 모두 current_chunk에 채웠을 때
      if 0 < len(current_chunk) :
        a_end = 1
        if 1 < len(current_chunk):
          a_end = random.randrange(1, len(current_chunk))
        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])
              
        tokens_b = []
        if len(current_chunk) == 1 or random.random() < 0.5 :
          is_next = 0
          tokens_b_len = tgt_seq - len(tokens_a)
          random_doc_idx = doc_idx
          while doc_idx == random_doc_idx :
            random_doc_idx = random.randrange(0, len(docs))
          random_doc = docs[random_doc_idx]

          random_start = random.randrange(0, len(random_doc))
          for j in range(random_start, len(random_doc)):
            tokens_b.extend(random_doc[j])
        
        else :
          is_next = 1
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        
        trim_tokens(tokens_a, tokens_b, max_seq)
        assert 0 < len(tokens_a)
        assert 0 < len(tokens_b)

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens)- 3) * mask_prob), vocab_list)

        instance = {"tokens" : tokens, "segment" : segment, "is_next" : is_next,                      "mask_idx" : mask_idx, "mask_label" : mask_label}
        instances.append(instance)

    current_chunk = []
    current_length = 0
  return instances



def make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob):
  vocab_list = []
  for id in range(vocab.get_piece_size()):
    if not vocab.is_unknown(id):
      vocab_list.append(vocab.id_to_piece(id))
  
  line_cnt = 0
  with open(in_file, 'r') as in_f :
    for line in in_f :
      line_cnt += 1

  docs = []
  with open(in_file, 'r') as f :
    doc = []
    with tqdm(total=line_cnt, desc=f"Loading") as pbar:
      for i, line in enumerate(f):
        if i >= 4296976*0.4 :
          break
        line = line.strip()
        if line == "":
          if 0 < len(doc):
            docs.append(doc)
            doc = []
        else :
          pieces = vocab.encode_as_pieces(line)
          if 0 < len(pieces):
            doc.append(pieces)
        pbar.update(1)
    if doc :
      docs.append(doc)

  for index in range(count):
    output = out_file.format(index)
    if os.path.isfile(output): continue

    with open(output, 'w') as out_f:
      with tqdm(total=len(docs), desc=f"Making") as pbar:
        for i, doc in enumerate(docs):
          instances = create_pretrain_instances(docs, i, doc, n_seq, mask_prob, vocab_list)
          for instance in instances:
            out_f.write(json.dumps(instance))
            out_f.write('\n')
          pbar.update(1)


if name == "__main__":

  vocab_file = "/content/drive/MyDrive/NLP/BERT/kowiki_8000.model"
  vocab = spm.SentencePieceProcessor()
  vocab.load(vocab_file)


  in_file = '/content/drive/MyDrive/NLP/BERT/kowiki.txt'
  out_file = '/content/drive/MyDrive/NLP/BERT/bert_kowiki_out_8000/kowiki_bert_8000_{}.json'
  count = 5
  n_seq = 256
  mask_prob = 0.15

  make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob)





