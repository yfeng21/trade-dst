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
import itertools

from .fix_label import *

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class UtteranceDataset(data.Dataset):
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
        turn_domain = self.map_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        generate_y = self.map_slot(generate_y, self.src_word2id)
        context = self.dialog_history[index] 
        context = self.map_utter(context, self.src_word2id)
        context_plain = self.dialog_history[index]
        
        item = {
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
        return item

    def __len__(self):
        return self.num_total_seqs
    
    def map_utter(self, sequence, word2idx: dict):
        """Converts words to ids."""
        story = [word2idx.get(word, UNK_token) for word in sequence.split()]
        return story

    def map_slot(self, sequence, word2idx:dict):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx.get(word, UNK_token) for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story
    #
    # def preprocess_memory(self, sequence, word2idx):
    #     """Converts words to ids."""
    #     story = []
    #     for value in sequence:
    #         d, s, v = value
    #         s = s.replace("book","").strip()
    #         # separate each word in value to different memory slot
    #         for wi, vw in enumerate(v.split()):
    #             idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
    #             story.append(idx)
    #     story = torch.Tensor(story)
    #     return story

    @staticmethod
    def map_domain(turn_domain):
        domains = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        return domains[turn_domain]


def collate_fn(data):
    def sent_to_batch(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = list(map(len, sequences))
        max_len = max(max(lengths), 1)

        padded = []
        for i, l in enumerate(lengths):
            padded_sent = sequences[i] + [PAD_token for _ in range(max_len - l)]
            padded.append(padded_sent)

        padded = torch.LongTensor(padded)
        return padded, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = [[len(s) for s in ss] for ss in sequences]
        max_len = max(sum(lengths, []))
        padded = []
        for ss in sequences:
            pp = []
            for s in ss:
                s = s + [PAD_token for _ in range(max_len-len(s))]
                pp.append(s)
            padded.append(pp)
        padded = torch.tensor(padded)
        lengths = torch.tensor(lengths)
        return padded, lengths

    # def merge_memory(sequences):
    #     lengths = [len(seq) for seq in sequences]
    #     max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
    #     padded_seqs = torch.ones(len(sequences), max_len, 4).long()
    #     for i, seq in enumerate(sequences):
    #         end = lengths[i]
    #         if len(seq) != 0:
    #             padded_seqs[i,:end,:] = seq[:end]
    #     return padded_seqs, lengths
  
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    batch = {}
    for key in data[0].keys():
        batch[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = sent_to_batch(batch['context'])
    y_seqs, y_lengths = merge_multi_response(batch["generate_y"])
    # gating_label = torch.tensor(batch["gating_label"])
    # turn_domain = torch.tensor(batch["turn_domain"])

    src_seqs = src_seqs.cuda()
    # gating_label = gating_label.cuda()
    # turn_domain = turn_domain.cuda()
    y_seqs = y_seqs.cuda()
    y_lengths = y_lengths.cuda()

    batch["context"] = src_seqs
    batch["context_len"] = src_lengths
    batch["generate_y"] = y_seqs
    batch["y_lengths"] = y_lengths

    batch["gating_label"] = torch.tensor(batch["gating_label"]).cuda()
    batch["turn_domain"] = torch.tensor(batch["turn_domain"]).cuda()

    return batch

def read_langs(args: argparse.Namespace, file_name, gating_dict, SLOTS, dataset, lang, training, max_line = None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {} 
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first 
        for dial_dict in dials:
            if (args.all_vocab or dataset=="train") and training:
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')

        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

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
                slot_temp = SLOTS
                # if dataset == "train" or dataset == "dev":
                #     if args["except_domain"] != "":
                #         slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                #         turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                #     elif args["only_domain"] != "":
                #         slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                #         turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                # else:
                #     if args["except_domain"] != "":
                #         slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                #         turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                #     elif args["only_domain"] != "":
                #         slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                #         turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

                class_label, generate_y, slot_mask, gating_label  = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys(): 
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

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
                
                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())
                
            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    print("domain_counter", domain_counter)
    return data, max_resp_len, slot_temp


def get_seq(pairs, lang, batch_size, shuffle):
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 

    dataset = UtteranceDataset(data_info, lang.word2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn)
    return data_loader



def prepare_data_seq(args):
    training = args.train
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
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    # Vocabulary
    lang = Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args.all_vocab else 'lang-train.pkl'


    with open(cache_path+lang_name, 'rb') as handle:
        lang = pickle.load(handle)
    pair_train, train_max_len, slot_train = read_langs(args, file_train, gating_dict, ALL_SLOTS, "train", lang, training)
    train = get_seq(pair_train, lang, train_batch_size, True)
    nb_train_vocab = lang.n_words
    pair_dev, dev_max_len, slot_dev = read_langs(args, file_dev, gating_dict, ALL_SLOTS, "dev", lang, training)
    dev   = get_seq(pair_dev, lang, eval_batch_size, False)
    pair_test, test_max_len, slot_test = read_langs(args, file_test, gating_dict, ALL_SLOTS, "test", lang, training)
    test  = get_seq(pair_test, lang, eval_batch_size, False)

    print("Read %s pairs train" % len(pair_train), flush=True)
    print("Read %s pairs dev" % len(pair_dev), flush=True)
    print("Read %s pairs test" % len(pair_test), flush=True)
    print("Vocab_size: %s " % lang.n_words, flush=True)
    print("Vocab_size Training %s" % nb_train_vocab , flush=True)
    print("USE_CUDA={}".format(USE_CUDA), flush=True)

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))), flush=True)
    print(SLOTS_LIST[2], flush=True)
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))), flush=True)
    print(SLOTS_LIST[3], flush=True)
    return train, dev, test, lang, SLOTS_LIST, gating_dict, nb_train_vocab
