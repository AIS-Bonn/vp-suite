import torch

def preprocess_img(x):
    '''
    [0, 255, np.uint8] -> [-1, 1, torch.float32]
    '''
    permutation = (2, 0, 1) if x.ndim == 3 else (0, 3, 1, 2)
    torch_x = torch.from_numpy(x.transpose(permutation).astype('float32'))
    #return (2 * torch_x / 255) - 1
    return torch_x / 255  # [0., 1.]

def postprocess_img(x):
    '''
    [~-1, ~1, torch.float32] -> [0, 255, np.uint8]
    '''
    # scaled_x = (torch.clamp(x, -1, 1) + 1) * 255 / 2
    scaled_x = torch.clamp(x, 0, 1) * 255  # from [0., 1.]
    return scaled_x.cpu().numpy().astype('uint8')
