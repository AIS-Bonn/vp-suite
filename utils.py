import numpy as np
import math, sys
from PIL import Image
import matplotlib.pyplot as plt
import hsluv

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


# helper function for data visualization
def save_vis(out_fp, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(out_fp)


def colorize_semseg(input):
    assert input.ndim == 3  # input: [h, w, num_object_classes]
    h, w, num_obj_classes = input.shape
    input = (input * np.arange(1, num_obj_classes+1)).sum(axis=-1).astype('uint8')

    colors = np.zeros((num_obj_classes+1, 3)).astype('uint8')
    colors[0] = [255, 255, 255]
    for i in range(num_obj_classes):
        hue = i * 360.0 / num_obj_classes
        rgb = hsluv.hsluv_to_rgb([hue, 100, 40])
        colors[i+1] = (np.array(rgb) * 255.0).astype('uint8')

    flattened = input.reshape(-1)  # N
    colorized = colors[flattened]  # N x 3
    return colorized.reshape(h, w, 3)