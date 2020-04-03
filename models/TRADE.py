import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim

import random
import numpy as np
import os
from tqdm import tqdm


class TRADE(nn.Module):
    def __init__(self, hidden_size, w2i, i2w, path, lr, dropout, slots, gating_dict, emb_path=None):
        super(TRADE, self).__init__()
        self.hidden_size = hidden_size
        self.w2i = w2i
        self.i2w = i2w
        self.lr = lr
        self.dropout = dropout
        self.slots = slots
        self.gating_dict = gating_dict
        self.num_gates = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()

        self.encoder = EncoderRNN(len(self.w2i), hidden_size, self.dropout, emb_path=emb_path)
        self.decoder = Generator(self.i2w, self.encoder.embedding, len(self.w2i), hidden_size, self.dropout,
                                 self.slots, self.num_gates)

        if path:
            self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.bin')))
            self.decoder.load_state_dict(torch.load(os.path.join(path, 'decoder.bin')))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.reset()
        self.encoder.cuda()
        self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self):
        directory = 'saved_model'
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder.state_dict(), os.path.join(directory, 'encoder.bin'))
        torch.save(self.decoder.state_dict(), os.path.join(directory, 'decoder.bin'))

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def run_batch(self, batch, slots, reset=False):
        if reset: self.reset()

        use_teacher_forcing = random.random() < 0.5
        all_point_outputs, gates, words_point_out, words_class_out = self.run_encoder_decoder(
            batch, use_teacher_forcing, slots)
        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            batch["generate_y"].contiguous(),
            batch["y_lengths"]
        )
        loss_gate = self.cross_entorpy(
            gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
            batch["gating_label"].contiguous().view(-1)
        )

        loss = loss_ptr + loss_gate

        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

        return loss

    def clip(self, clip):
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

    def run_encoder_decoder(self, batch, teacher_forcing, slots):
        if True and self.decoder.training:
            story_shape = batch['context'].shape
            rand_mask = torch.rand(*story_shape) > self.dropout
            story = batch['context'] * rand_mask.long().cuda()
        else:
            story = batch['context']

        encoder_outputs, encoder_hidden = self.encoder(
            story.transpose(0, 1), batch['context_len'])

        batch_size = len(batch['context_len'])
        self.copy_list = batch['context_plain']
        max_res_len = batch['generate_y'].size(2) if self.encoder.training else 10
        all_outputs = self.decoder(
            batch_size, encoder_hidden, encoder_outputs, batch['context_len'], story, max_res_len, batch['generate_y'],
            teacher_forcing, slots)
        return all_outputs

    def evaluate(self, eval_data, best_joint, slots):
        self.encoder.eval()
        self.decoder.eval()
        print("EVALUATION")
        model_predictions = {}
        inverse_gating_dict = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(eval_data), total=len(eval_data))
        for j, eval_data in pbar:
            batch_size = len(eval_data['context_len'])
            _, gates, words_point_out, words_class_out = self.run_encoder_decoder(eval_data, False, slots)

            for b in range(batch_size):
                if eval_data["ID"][b] not in model_predictions.keys():
                    model_predictions[eval_data["ID"][b]] = {}
                model_predictions[eval_data["ID"][b]][eval_data["turn_id"][b]] = {
                    "turn_belief": eval_data["turn_belief"][b]}
                b_model_prediction = []
                gate = torch.argmax(gates.transpose(0, 1)[b], dim=1)

                for slot_index, gate_index in enumerate(gate):
                    if gate_index == self.gating_dict["none"]:
                        continue
                    elif gate_index == self.gating_dict["ptr"]:
                        pred = np.transpose(words_point_out[slot_index])[b]
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
                            b_model_prediction.append(slots[slot_index] + "-" + str(st))
                    else:
                        b_model_prediction.append(slots[slot_index] + "-" + inverse_gating_dict[gate_index.item()])
                model_predictions[eval_data["ID"][b]][eval_data["turn_id"][b]]["pred_bs_ptr"] = b_model_prediction

        joint_acc, slot_acc = self.evaluate_metrics(model_predictions, slots)
        print("Joint Acc:{:.4f}, slot Acc:{:.4f}".format(joint_acc,slot_acc))

        self.encoder.train()
        self.decoder.train()

        if (joint_acc >= best_joint):
            self.save_model()
            print("Best model with joint accuracy={:.4f} saved".format(joint_acc))
        return joint_acc


    def evaluate_metrics(self, predictions, slots):
        total, slot_acc, joint_acc = 0, 0, 0
        for _, pred in predictions.items():
            for i in range(len(pred)):
                curr_pred = pred[i]
                if set(curr_pred["turn_belief"]) == set(curr_pred["pred_bs_ptr"]):
                    joint_acc += 1
                total += 1
                temp_acc = self.compute_acc(set(curr_pred["turn_belief"]), set(curr_pred["pred_bs_ptr"]), slots)
                slot_acc += temp_acc

        joint_acc_score = joint_acc / float(total) if total != 0 else 0
        slot_acc_score = slot_acc / float(total) if total != 0 else 0
        return joint_acc_score, slot_acc_score

    def compute_acc(self, gold, predictions, slots):
        missed = 0
        missed_slots = []
        for g in gold:
            if g not in predictions:
                missed += 1
                missed_slots.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in predictions:
            if p not in gold and p.rsplit("-", 1)[0] not in missed_slots:
                wrong_pred += 1
        acc = len(slots) - missed - wrong_pred
        return acc / float(len(slots))


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
                padding_idx=1
            )
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=1)
            nn.init.normal_(self.embedding.weight, 0., 0.1)

    def forward(self, input_seqs, input_lengths=None, hidden=None):
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
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, i2w, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.i2w = i2w
        self.word_embedding = shared_emb
        self.dropout = nn.Dropout(dropout)
        self.num_gates = nb_gate
        self.hidden_size = hidden_size
        self.slots = slots

        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.interpolation = nn.Linear(3 * hidden_size, 1)
        self.gating = nn.Linear(hidden_size, nb_gate)
        self.slot_w2i = {'UNK_SLOT': 0}
        for slot in self.slots:
            for s in slot.split("-")[:2]:
                if s not in self.slot_w2i:
                    self.slot_w2i[s] = len(self.slot_w2i)
        self.slot_embedding = nn.Embedding(len(self.slot_w2i), hidden_size, padding_idx=0)
        nn.init.normal_(self.slot_embedding.weight)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches,
                use_teacher_forcing, candidate_slots):
        domain_ids = list(map(lambda x: self.slot_w2i.get(x.split("-")[0], 0), candidate_slots))
        slot_ids = list(map(lambda x: self.slot_w2i.get(x.split("-")[1], 0), candidate_slots))
        domain_ids = torch.LongTensor(domain_ids).cuda()
        slot_ids = torch.LongTensor(slot_ids).cuda()
        slot_embs = self.slot_embedding(domain_ids) + self.slot_embedding(slot_ids)

        all_point_outputs = []
        num_slots = len(candidate_slots)
        _, seq_len, _ = encoded_outputs.shape
        decoder_input = self.dropout(slot_embs).unsqueeze(1).expand(-1, batch_size, self.hidden_size)  # S * B * D
        decoder_input = decoder_input.reshape(-1, self.hidden_size)

        expanded_enocder_hidden = encoded_hidden.squeeze().repeat(num_slots, 1)
        hidden = expanded_enocder_hidden.unsqueeze(0)

        expanded_encoder_outputs = encoded_outputs.repeat(num_slots, 1, 1)

        expanded_encoder_lengths = encoded_lens * num_slots
        words = []
        for position in range(max_res_len):
            dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

            context_vec, logits, prob = self.attend(
                expanded_encoder_outputs, hidden.squeeze(0), expanded_encoder_lengths)
            if position == 0:
                all_gate_outputs = self.gating(context_vec).view((len(candidate_slots), batch_size, self.num_gates))
            p_vocab = torch.matmul(hidden.squeeze(0), self.word_embedding.weight.transpose(1, 0))
            p_vocab = F.softmax(p_vocab, dim=1)

            p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)

            interp = F.sigmoid(self.interpolation(p_gen_vec))
            p_context_ptr = torch.zeros(p_vocab.size()).cuda()
            p_context_ptr.scatter_add_(1, story.repeat(num_slots, 1), prob)

            final_p_vocab = (1 - interp) * p_context_ptr + interp * p_vocab

            pred_word = torch.argmax(final_p_vocab, dim=1)
            words.append(pred_word.view(num_slots, batch_size).cpu())
            all_point_outputs.append(final_p_vocab.view(num_slots, batch_size, -1))
            if use_teacher_forcing:
                decoder_input = self.word_embedding(
                    target_batches[:, :, position].transpose(0, 1).reshape(-1))  # Chosen word is next input
            else:
                decoder_input = self.word_embedding(pred_word)
            decoder_input = decoder_input.cuda()
        all_point_outputs = torch.stack(all_point_outputs, dim=2)
        words = torch.stack(words, dim=1).tolist()
        words = [[[self.i2w[z] for z in y] for y in x] for x in words]
        return all_point_outputs, all_gate_outputs, words, []

    def attend(self, hiddens, query, lengths):
        unnormalized_scores = torch.bmm(hiddens, query.unsqueeze(-1)).squeeze(-1)
        max_len = max(lengths)
        for i, l in enumerate(lengths):
            if l < max_len:
                unnormalized_scores.data[i, l:] = -np.inf
        scores = F.softmax(unnormalized_scores, dim=1)  # B * L
        context = torch.bmm(scores.unsqueeze(1), hiddens).squeeze(1)
        return context, unnormalized_scores, scores

def masked_cross_entropy_for_value(logits, target, mask):
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size()) # b * |s| * m
    loss = masking(losses, mask)
    return loss

def masking(losses, mask):
    batch_size, num_slots, seq_len = losses.shape
    seq_mask = torch.arange(0, seq_len).long().unsqueeze(0).unsqueeze(0).expand(batch_size, num_slots, -1)  # B * S * L
    seq_mask = seq_mask.cuda() < mask.unsqueeze(-1)
    losses = losses * seq_mask.float()
    loss = losses.sum() / (seq_mask.sum().float())
    return loss