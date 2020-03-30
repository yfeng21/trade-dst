import torch

def masked_cross_entropy_for_value(logits, target, mask):
    # logits: b * |s| * m * |v|
    # target: b * |s| * m
    # mask:   b * |s|
    logits_flat = logits.view(-1, logits.size(-1)) ## -1 means infered from other dimentions
    # print(logits_flat.size())
    log_probs_flat = torch.log(logits_flat)
    # print("log_probs_flat", log_probs_flat)
    target_flat = target.view(-1, 1)
    # print("target_flat", target_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size()) # b * |s| * m
    loss = masking(losses, mask)
    return loss

def masking(losses, mask):
    # mask_ = []
    # batch_size = mask.size(0)
    # seq_len = losses.size(2)
    batch_size, num_slots, seq_len = losses.shape
    seq_mask = torch.arange(0, seq_len).long().unsqueeze(0).unsqueeze(0).expand(batch_size, num_slots, -1)  # B * S * L
    seq_mask = seq_mask.cuda() < mask.unsqueeze(-1)
    # for si in range(mask.size(1)):
    #     seq_range = torch.arange(0, seq_len).long()
    #     seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, seq_len)
    #     if mask[:,si].is_cuda:
    #         seq_range_expand = seq_range_expand.cuda()
    #     seq_length_expand = mask[:, si].unsqueeze(1).expand_as(seq_range_expand)
    #     mask_.append( (seq_range_expand < seq_length_expand) )  # B * L
    # mask_ = torch.stack(mask_)  # S * B * L
    # mask_ = mask_.transpose(0, 1)  # B * S * L
    # if losses.is_cuda:
    #     mask_ = mask_.cuda()
    losses = losses * seq_mask.float()
    loss = losses.sum() / (seq_mask.sum().float())
    return loss



