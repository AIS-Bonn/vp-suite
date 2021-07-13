import numpy as np
import math
from PIL import Image

def get_grid_vis(input, mode='RGB'):

    val = input.detach().clone()
    b, c, h, w = val.shape
    grid_size = math.ceil(math.sqrt(b))
    imgmatrix = np.zeros((3 if mode == "RGB" else 1,
                          (h+1) * grid_size - 1,
                          (w+1) * grid_size - 1,))

    for i in range(b):
        x, y = i % grid_size, i // grid_size
        imgmatrix[:, y * (h+1) : (y+1)*(h+1)-1, x * (w+1) : (x+1)*(w+1)-1] = val[i]

    imgmatrix = imgmatrix.transpose((1, 2, 0)) if mode == "RGB" else imgmatrix.squeeze()
    return Image.fromarray((imgmatrix * 255).astype('uint8')).convert("RGB")