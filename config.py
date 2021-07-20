import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 7
IMG_SIZE = (256, 256)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
LOAD_MODEL = False

VIDEO_IN_LENGTH = 6
VIDEO_PRED_LENGTH = 3