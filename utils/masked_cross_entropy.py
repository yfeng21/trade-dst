import torch

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



