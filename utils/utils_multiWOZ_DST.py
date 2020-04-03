import json
import torch
from utils.config import *
import pickle

from .fix_label import *


class UtteranceDataset(torch.utils.data.Dataset):
    def __init__(self, data_info, src_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.generate_y = data_info["generate_y"]
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        generate_y = self.generate_y[index]
        generate_y = self.map_slot(generate_y, self.src_word2id)
        context = self.dialog_history[index]
        context = self.map_utter(context, self.src_word2id)
        context_plain = self.dialog_history[index]

        item = {
            "ID": ID,
            "turn_id": turn_id,
            "turn_belief": turn_belief,
            "gating_label": gating_label,
            "context": context,
            "context_plain": context_plain,
            "generate_y": generate_y,
        }
        return item

    def __len__(self):
        return self.num_total_seqs

    def map_utter(self, sequence, word2idx: dict):
        """Converts words to ids."""
        story = [word2idx.get(word, UNK_token) for word in sequence.split()]
        return story

    def map_slot(self, sequence, word2idx: dict):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx.get(word, UNK_token) for word in value.split()] + [EOS_token]
            story.append(v)
        return story


def collate_fn(data):
    def sent_to_batch(sequences):
        lengths = list(map(len, sequences))
        max_len = max(max(lengths), 1)

        padded = []
        for i, l in enumerate(lengths):
            padded_sent = sequences[i] + [PAD_token for _ in range(max_len - l)]
            padded.append(padded_sent)

        padded = torch.LongTensor(padded)
        return padded, lengths

    def slot_values_to_batch(sequences):
        lengths = [[len(s) for s in ss] for ss in sequences]
        max_len = max(sum(lengths, []))
        padded = []
        for ss in sequences:
            pp = []
            for s in ss:
                s = s + [PAD_token for _ in range(max_len - len(s))]
                pp.append(s)
            padded.append(pp)
        padded = torch.tensor(padded)
        lengths = torch.tensor(lengths)
        return padded, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True)
    batch = {}
    for key in data[0].keys():
        batch[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = sent_to_batch(batch['context'])
    y_seqs, y_lengths = slot_values_to_batch(batch["generate_y"])

    src_seqs = src_seqs.cuda()
    y_seqs = y_seqs.cuda()
    y_lengths = y_lengths.cuda()

    batch["context"] = src_seqs
    batch["context_len"] = src_lengths
    batch["generate_y"] = y_seqs
    batch["y_lengths"] = y_lengths

    batch["gating_label"] = torch.tensor(batch["gating_label"]).cuda()

    return batch


def load_dialog_file(file_name, gating_dict, SLOTS, w2i, batch_size, shuffle):
    print(("Reading from {}".format(file_name)))
    data = []
    with open(file_name) as f:
        dials = json.load(f)
        for dial_dict in dials:
            dialog_history = ""
            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_id = turn["turn_idx"]
                dialog_history += (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)
                turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]
                class_label, generate_y, slot_mask, gating_label = [], [], [], []
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
                    "ID": dial_dict["dialogue_idx"],
                    "turn_id": turn_id,
                    "dialog_history": source_text,
                    "turn_belief": turn_belief_list,
                    "gating_label": gating_label,
                    'generate_y': generate_y
                }
                data.append(data_detail)

    data_info = {}
    data_keys = data[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in data:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = UtteranceDataset(data_info, w2i)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)
    return data_loader


def prepare_data_seq(args):
    train_batch_size = args.train_batch_size
    eval_batch_size = args.train_batch_size
    file_train = args.data_dir + '/train_dials.json'
    file_dev = args.data_dir + '/dev_dials.json'
    file_test = args.data_dir + '/test_dials.json'
    gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
    with open(args.data_dir + '/all_slots.pkl', 'rb') as f:
        candidate_slots = pickle.load(f)
    with open(args.data_dir + '/vocab_dict.pkl', 'rb') as handle:
        i2w = pickle.load(handle)
        w2i = pickle.load(handle)
    train = load_dialog_file(file_train, gating_dict, candidate_slots, w2i, train_batch_size, True)
    dev = load_dialog_file(file_dev, gating_dict, candidate_slots, w2i, eval_batch_size, False)
    test = load_dialog_file(file_test, gating_dict, candidate_slots, w2i, eval_batch_size, False)
    return train, dev, test, w2i, i2w, candidate_slots, gating_dict
