import sys, os, time
from pathlib import Path

import torch.nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from dataset import SynpickVideoDataset
from models.prediction.pred_model import CopyLastFrameModel, UNet3d
from utils import validate_video_model
from visualize import visualize_video

def main(args):

    # DATA
    data_dir = args[0]
    train_dir = os.path.join(data_dir, 'train', 'rgb')
    val_dir = os.path.join(data_dir, 'val', 'rgb')
    test_dir = os.path.join(data_dir, 'test', 'rgb')

    train_data = SynpickVideoDataset(data_dir=train_dir, sequence_length=VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH)
    val_data = SynpickVideoDataset(data_dir=val_dir, sequence_length=VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    # MODEL
    # pred_model = CopyLastFrameModel()
    pred_model = UNet3d(in_channels=3, out_channels=3, time_dim=VIDEO_IN_LENGTH).to(DEVICE)

    # ETC
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=LEARNING_RATE)

    min_loss = float("inf")
    timestamp = int(1000000 * time.time())
    out_dir = Path("out/{}_pred_model".format(timestamp))
    out_dir.mkdir(parents=True)

    # TRAINING
    for i in range(0, NUM_EPOCHS):
        print('\nEpoch: {}'.format(i))

        loop = tqdm(train_loader)
        for batch_idx, data in enumerate(loop):
            data = data.to(DEVICE)  # [b, T, h, w], with T = in_length + pred_length
            input, targets = data[:, :VIDEO_IN_LENGTH], data[:, VIDEO_IN_LENGTH:]

            predictions = pred_model.pred_n(input, pred_length=VIDEO_PRED_LENGTH)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        print("Validating...")
        cur_loss = validate_video_model(valid_loader, pred_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, loss_fn)
        print("Val. MSE loss = {}".format(cur_loss))

        # save model if last epoch improved acc.
        if min_loss > cur_loss:
            max_accuracy = cur_loss
            torch.save(pred_model, str((out_dir / 'best_model.pth').resolve()))
            print('Model saved!')

        # visualize model predictions using eval mode and validation data
        visualize_video(val_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, pred_model, out_dir)

        if i == 25:
            optimizer.param_groups[0]['lr'] *= 0.1
            print('Decrease learning rate!')

    # TESTING
    print("\nTraining done, testing best model...")
    best_model = torch.load(str((out_dir / 'best_model.pth').resolve()))
    test_data = SynpickVideoDataset(data_dir=test_dir, sequence_length=VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    test_loss = validate_video_model(test_loader, best_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, loss_fn)
    print("Test MSE loss = {}".format(test_loss))

    visualize_video(test_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, best_model, out_dir)
    print("Testing done, bye bye!")


if __name__ == '__main__':
    main(sys.argv[1:])