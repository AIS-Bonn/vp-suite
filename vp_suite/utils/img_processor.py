import torch

class ImgProcessor():
    def __init__(self, value_min, value_max):
        self.value_min, self.value_max = value_min, value_max

    def preprocess_img(self, x):
        '''
        Converts a numpy array describing an image to its normalized tensor representation.
        Input: np.uint8, shape: [..., h, w, c], range: [0, 255]
        Output: torch.float, shape: [..., c, h, w], range: [min_val, max_val]
        '''
        permutation = (2, 0, 1) if x.ndim == 3 else (0, 3, 1, 2)
        x = torch.from_numpy(x.transpose(permutation).astype('float32'))
        x = x / 255.  # [0, 1]
        x = x * (self.value_max - self.value_min)  # [0, max_val - min_val]
        x = x + self.value_min  #  [min_val, max_val]
        return x

    def postprocess_img(self, x):
        '''
        Converts a normalized tensor of an image to a denormalized numpy array.
        Input: torch.float, shape: [..., c, h, w], range (approx.): [min_val, max_val]
        Output: np.uint8, shape: [..., h, w, c], range: [0, 255]
        '''
        permutation = (1, 2, 0) if x.ndim == 3 else (0, 2, 3, 1)
        x = x - self.value_min  # ~[0, max_val - min_val]
        x = x / (self.value_max - self.value_min)  # ~[0, 1]
        x = x * 255.  # ~[0, 255]
        x = torch.clamp(x, 0., 255.)
        x = x.cpu().numpy().astype('uint8').transpose(permutation)
        return x
