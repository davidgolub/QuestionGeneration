import torch 
from torch.autograd import variable
from helpers import constants 
import models
import functools
import numpy as np 
import torch.nn.utils.rnn as rnn_utils
from copy import deepcopy 
VERY_LARGE_NEGATIVE_NUMBER = 0.0 # TODO: FIGURE OUT MASKING

def set_gpu(tensor, gpu_mode=False):
    if gpu_mode:
        tensor = tensor.cuda()
    return tensor

def reshape_forward(tensor, module):
    """
    Reshapes 3d tensor to 2d one when forwarding 
    """
    tensor_size = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_size[-1])
    reshaped_logits = module(reshaped_tensor)
    logits = reshaped_logits.view(tensor_size[0], tensor_size[1], -1)
    return logits 

def num_elements(tensor):
    return functools.reduce(lambda x, y: x * y, tensor.size())

def average_accuracy(t1, t2):
    """
    t1: Torch tensor  
    t2: Torch tensor
    """
    return torch.sum(torch.eq(t1, t2)) / num_elements(t1)

def to_bool(tensor):
    """
    tensor: torch tensor with one value 
    returns boolean value of tensor
    """
    return (tensor.cpu() > 0).numpy()

def get_index_select(masks):
    """
    Get index select tensor from a list of masks
    """
    num_rows = masks.size(0)
    num_cols =  masks.size(1)
    new_tensor = []
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            if to_bool(masks[i][j].data.cpu() == torch.LongTensor([0])):
                new_tensor.append(i * num_cols + j)

    indices = torch.from_numpy(np.array(new_tensor)).long()
    flattened_indices = variable.Variable(indices)
    return flattened_indices 

def sort_sequence(tensor, lengths, batch_first=False):
    """
    Sorts sequence in descending order
    tensor: Padded tensor of variable length stuff (Torch tensor)
    lengths: Lengths of padded tensor (Torch LongTensor)
    batch_first: Boolean, whether tensor is batch_first or not  
    """
    idx = None
    if batch_first:
        idx = 0
    else:
        idx = 1 

    sorted_lengths, indices = torch.sort(lengths, dim=0, descending=True)
    new_tensor = torch.index_select(tensor, idx, indices)
    return new_tensor, sorted_lengths, indices

def unsort_sequence(tensor, indices, batch_first=False):
    """
    Unsort a tensor according to indices and idx
    """
    if batch_first:
        idx = 0
    else:
        idx = 1 
    unsorted_tensor = torch.index_select(tensor, idx, indices)
    return unsorted_tensor

def pack_forward(rnn, h0, tensor, lengths, batch_first=False):
    """
    Forwards a padded tensor with lengths lengths thru rnn 
    rnn: Cell to forward through 
    h0: initial states of cell: tuple (h0, c0)
    tensor: Tensor to use 
    lengths: Lengths to use 
    batch_first: Whether tensor is batch first or not
    """

    sorted_tensor, sorted_lengths, sorted_indices = sort_sequence(tensor, lengths, batch_first)
    packed = rnn_utils.pack_padded_sequence(sorted_tensor, sorted_lengths.data.cpu().numpy())
    packed_out, packed_hidden = rnn(packed, h0)
    unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed_out)
    unsorted_out = unsort_sequence(unpacked, sorted_indices, batch_first=False)
    unsorted_hidden = list(map(lambda idx: unsort_sequence(packed_hidden[idx], sorted_indices, batch_first=False), [0, 1]))
    return unsorted_out, unsorted_hidden

def save_model(model, path, additional_args=None):
    save_dict = {}
    if additional_args is not None:
        save_dict.update(additional_args)
    set_gpu(model, False)
    save_dict['state_dict'] = model.state_dict()
    save_dict['config'] = deepcopy(model.config)
    save_dict['config']['gpu_mode'] = False
    save_dict['type'] = model.get_type()

    torch.save(save_dict, path)
    set_gpu(model, model.config['gpu_mode'])

def get_model(model_type):
    if model_type == constants.MODEL_TYPE_LANGUAGE_MODEL:
        return models.language_model.LanguageModel 
    else:
        raise Exception("Invalid model type given %s" % model_type)

def load_model(path):
    print("Loading model from path %s" % path)
    data = torch.load(path)
    print("Done loading model from path %s" % path)
    config = data['config']
    model_type = get_model(data['type'])
    model = model_type(config)
    model.load_state_dict(data['state_dict'])
    return model 

def mask(tensor, mask):
    masked_tensor = tensor + mask.float() * VERY_LARGE_NEGATIVE_NUMBER
    return masked_tensor



