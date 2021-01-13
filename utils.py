import torch
import torchaudio
import torchvision
import torch.nn as nn

#Implementation from https://github.com/CannyLab/aai/blob/4a93c14d834f045ee3fa61929c4f17ebc765d10c/aai/utils/torch/masking.py#L20
def get_src_conditional_mask(max_sequence_length):
    mask = (torch.triu(torch.ones(max_sequence_length, max_sequence_length)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

def sequence_mask(lengths, maxlen=None, right_aligned=False):
    # https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036
    if maxlen is None:
        maxlen = lengths.max()
    matrix = torch.unsqueeze(lengths, dim=-1)
    row_vector = torch.arange(0, maxlen, 1).type_as(matrix)
    if not right_aligned:
        mask = row_vector < matrix
    else:
        mask = row_vector > (-matrix + (maxlen - 1))

    return mask.bool()