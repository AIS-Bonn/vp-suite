import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SYNPICK_CLASSES = ['object_{}'.format(i) for i in range(1, 22)]

BATCH_SIZE = 16
IMG_SIZE = (256, 256)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
LOAD_MODEL = False
