import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
import json
# import pandas as pd
import copy

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
import pprint


class TRADE(nn.Module):
    def __init__(self, hidden_size, path, lr, dropout, slots, gating_dict, w2i, i2w, emb_path=None,):
        super(TRADE, self).__init__()
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()
        self.w2i = w2i
        self.i2w = i2w

        self.encoder = EncoderRNN(len(self.w2i), hidden_size, self.dropout, emb_path=emb_path)
        self.decoder = Generator(self.encoder.embedding, len(self.w2i), hidden_size, self.dropout,
                                 self.slots, self.nb_gate,self.i2w)

        if path:
            self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.bin')))
            self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.bin')))

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self, dec_type):
        directory = 'saved_model'
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder.state_dict(), os.path.join(directory, 'encoder.bin'))
        torch.save(self.decoder.state_dict(), os.path.join(directory, 'decoder.bin'))

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < 0.5
        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(
            data, use_teacher_forcing, slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(),  # [:,:len(self.point_slots)].contiguous(),
            data["y_lengths"])  # [:,:len(self.point_slots)])
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
                                       data["gating_label"].contiguous().view(-1))


        loss = loss_ptr + loss_gate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):
        # Build unknown mask for memory to encourage generalization
        if True and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA:
                rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
                                                                                                     encoded_hidden,
                                                                                                     encoded_outputs,
                                                                                                     data[
                                                                                                         'context_len'],
                                                                                                     story, max_res_len,
                                                                                                     data['generate_y'], \
                                                                                                     use_teacher_forcing,
                                                                                                     slot_temp)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                for si, sg in enumerate(gate):
                    if sg == self.gating_dict["none"]:
                        continue
                    elif sg == self.gating_dict["ptr"]:
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS':
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
                    else:
                        predict_belief_bsz_ptr.append(slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr",
                                                                                      slot_temp)

        evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                              "Joint F1": F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr  # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        if (joint_acc_score >= matric_best):
            self.save_model("best_model")
            print("MODEL SAVED ACC-{:.4f}".format(joint_acc_score))
        return joint_acc_score

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total != 0 else 0
        turn_acc_score = turn_acc / float(total) if total != 0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            if len(pred) == 0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1, emb_path=None):
        import pickle
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

        if emb_path:
            with open(emb_path, 'rb') as f:
                weights = pickle.load(f)
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(weights),
                freeze=False,
                padding_idx=PAD_token
            )
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
            nn.init.normal_(self.embedding.weight, 0., 0.1)

    def forward(self, input_seqs, input_lengths=None, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout(embedded)
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden.sum(dim=0)
        seq_len, batch_size, _ = outputs.shape
        outputs = outputs.view(seq_len, batch_size, 2, -1).sum(-2)
        # outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate, i2w):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.i2w = i2w

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {'UNK_SLOT': 0}
        for slot in self.slots:
            for s in slot.split("-")[:2]:
                if s not in self.slot_w2i:
                    self.slot_w2i[s] = len(self.slot_w2i)
        self.slot_embedding = nn.Embedding(len(self.slot_w2i), hidden_size, padding_idx=0)
        nn.init.normal_(self.slot_embedding.weight)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches,
                use_teacher_forcing, slot_temp):
        # all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        # all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        # if USE_CUDA:
        #     all_point_outputs = all_point_outputs.cuda()
        #     all_gate_outputs = all_gate_outputs.cuda()

        # Get the slot embedding
        domain_ids = list(map(lambda x: self.slot_w2i.get(x.split("-")[0], 0), slot_temp))
        slot_ids = list(map(lambda x: self.slot_w2i.get(x.split("-")[1], 0), slot_temp))
        domain_ids = torch.LongTensor(domain_ids).cuda()
        slot_ids = torch.LongTensor(slot_ids).cuda()
        slot_embs = self.slot_embedding(domain_ids) + self.slot_embedding(slot_ids)

        all_point_outputs = []
        num_slots = len(slot_temp)
        _, seq_len, _ = encoded_outputs.shape
        decoder_input = self.dropout_layer(slot_embs).unsqueeze(1).expand(-1, batch_size, self.hidden_size)  # S * B * D
        decoder_input = decoder_input.reshape(-1, self.hidden_size)

        # expanded_enocder_hidden = encoded_hidden.unsqueeze(1).expand(1, num_slots, -1, -1)  # 1 * S * B * D
        # hidden = expanded_enocder_hidden.reshape(1, -1, self.hidden_size)  # 1 * (S*B) * D
        expanded_enocder_hidden = encoded_hidden.squeeze().repeat(num_slots, 1)
        hidden = expanded_enocder_hidden.unsqueeze(0)

        # expanded_encoder_outputs = encoded_outputs.unsqueeze(0).expand(num_slots, -1, -1, -1)  # S * B * L * D
        # expanded_encoder_outputs = expanded_encoder_outputs.reshape(-1, seq_len, self.hidden_size)  # (S * B) * L * D
        expanded_encoder_outputs = encoded_outputs.repeat(num_slots, 1, 1)

        expanded_encoder_lengths = encoded_lens * num_slots
        words = []
        for position in range(max_res_len):
            dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

            context_vec, logits, prob = self.attend(
                expanded_encoder_outputs, hidden.squeeze(0), expanded_encoder_lengths)
            if position == 0:
                all_gate_outputs = self.W_gate(context_vec).view((len(slot_temp), batch_size, self.nb_gate))
            p_vocab = torch.matmul(hidden.squeeze(0), self.embedding.weight.transpose(1, 0))
            p_vocab = F.softmax(p_vocab, dim=1)

            p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)

            interp = self.sigmoid(self.W_ratio(p_gen_vec))
            p_context_ptr = torch.zeros(p_vocab.size()).cuda()
            p_context_ptr.scatter_add_(1, story.repeat(num_slots, 1), prob)

            final_p_vocab = (1 - interp) * p_context_ptr + interp * p_vocab

            pred_word = torch.argmax(final_p_vocab, dim=1)
            words.append(pred_word.view(num_slots, batch_size).cpu())
            all_point_outputs.append(final_p_vocab.view(num_slots, batch_size, -1))
            if use_teacher_forcing:
                decoder_input = self.embedding(target_batches[:, :, position].transpose(0,1).reshape(-1))  # Chosen word is next input
            else:
                decoder_input = self.embedding(pred_word)
            decoder_input = decoder_input.cuda()
        all_point_outputs = torch.stack(all_point_outputs, dim=2)
        # words_point_out = list(map(list, zip(*words)))
        words = torch.stack(words, dim=1).tolist()
        words = [[[self.i2w[z] for z in y] for y in x] for x in words]
        return all_point_outputs, all_gate_outputs, words, []

    def attend(self, hiddens, query, lengths):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = query.unsqueeze(1).expand_as(hiddens).mul(hiddens).sum(2)
        # scores_ = torch.bmm(seq, cond.unsqueeze(-1)).squeeze(-1)
        max_len = max(lengths)
        for i, l in enumerate(lengths):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)  # B * L
        context = scores.unsqueeze(2).expand_as(hiddens).mul(hiddens).sum(1)
        # context = torch.bmm(scores.unsqueeze(1), seq).squeeze(1)
        return context, scores_, scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
