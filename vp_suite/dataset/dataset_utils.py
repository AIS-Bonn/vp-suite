import torch

def preprocess_img(x, min_val=-1.0, max_val=1.0):
    '''
    Converts a numpy array describing an image to its normalized tensor representation.
    Input: np.uint8, shape: [..., h, w, c], range: [0, 255]
    Output: torch.float, shape: [..., c, h, w], range: [min_val, max_val]
    '''
    permutation = (2, 0, 1) if x.ndim == 3 else (0, 3, 1, 2)
    x = torch.from_numpy(x.transpose(permutation).astype('float32'))
    x = x / 255.  # [0, 1]
    x = x * (max_val - min_val)  # [0, max_val - min_val]
    x = x + min_val  #  [min_val, max_val]
    return x

def postprocess_img(x, min_val=-1.0, max_val=1.0):
    '''
    Converts a normalized tensor of an image to a denormalized numpy array.
    Input: torch.float, shape: [..., c, h, w], range (approx.): [min_val, max_val]
    Output: np.uint8, shape: [..., h, w, c], range: [0, 255]
    '''
    permutation = (1, 2, 0) if x.ndim == 3 else (0, 2, 3, 1)
    x = x - min_val  # ~[0, max_val - min_val]
    x = x / (max_val - min_val)  # ~[0, 1]
    x = x * 255.  # ~[0, 255]
    x = torch.clamp(x, 0., 255.)
    x = x.cpu().numpy().astype('uint8').transpose(permutation)
    return x
