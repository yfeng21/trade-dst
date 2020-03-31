import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
from tqdm import tqdm
import os
import pickle
from random import shuffle

from .fix_label import *

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index] 
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]
        
        item_info = {
            "ID":ID, 
            "turn_id":turn_id, 
            "turn_belief":turn_belief, 
            "gating_label":gating_label, 
            "context":context, 
            "context_plain":context_plain, 
            "turn_uttr_plain":turn_uttr, 
            "turn_domain":turn_domain, 
            "generate_y":generate_y, 
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        return domains[turn_domain]


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths
  
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    return item_info

def load_dialog_file(file_name, gating_dict, SLOTS,w2i,batch_size):
    print(("Reading from {}".format(file_name)))
    data = []
    with open(file_name) as f:
        dials = json.load(f)

        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)

                # Generate domain-dependent slot list
                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]


                class_label, generate_y, slot_mask, gating_label  = [], [], [], []
                for slot in SLOTS:
                    if slot in turn_belief_dict.keys(): 
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])
                
                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "domains":dial_dict["domains"], 
                    "turn_domain":turn_domain,
                    "turn_id":turn_id, 
                    "dialog_history":source_text, 
                    "turn_belief":turn_belief_list,
                    "gating_label":gating_label, 
                    "turn_uttr":turn_uttr_strip, 
                    'generate_y':generate_y
                    }
                data.append(data_detail)

    data_info = {}
    data_keys = data[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in data:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, w2i)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


def process_data_dir(args):
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    file_train = args.data_dir+'/train_dials.json'
    file_dev = args.data_dir+'/dev_dials.json'
    file_test = args.data_dir+'/test_dials.json'
    # Create saving folder
    cache_path = args.cache_path + '/'
    print("caching file to", cache_path, flush=True)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    # load domain-slot pairs from ontology
    with open(args.data_dir+'/all_slots.pkl','rb') as f:
        ALL_SLOTS = pickle.load(f)
    with open(args.data_dir+'/vocab_dict.pkl','rb') as f:
        i2w = pickle.load(f)
        w2i = pickle.load(f)

    train = load_dialog_file(file_train, gating_dict, ALL_SLOTS, w2i, train_batch_size)
    dev = load_dialog_file(file_dev, gating_dict, ALL_SLOTS, w2i, eval_batch_size)
    test = load_dialog_file(file_test, gating_dict, ALL_SLOTS, w2i, eval_batch_size)
    return train, dev, test, ALL_SLOTS, gating_dict, i2w, w2i
