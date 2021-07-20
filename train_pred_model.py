import sys, os, time, argparse
from pathlib import Path

import torch.nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from dataset import SynpickVideoDataset
from models.prediction.pred_model import CopyLastFrameModel, UNet3d, LSTMModel
from metrics.fvd import FrechetVideoDistance
from utils import validate_video_model
from visualize import visualize_video

def main(args):

    # DATA
    data_dir = cfg.in_path
    train_dir = os.path.join(data_dir, 'train', 'rgb')
    val_dir = os.path.join(data_dir, 'val', 'rgb')
    test_dir = os.path.join(data_dir, 'test', 'rgb')


    train_data = SynpickVideoDataset(data_dir=train_dir, num_frames=VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH, step=3)
    val_data = SynpickVideoDataset(data_dir=val_dir, num_frames=VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH, step=3)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    # MODEL
    if cfg.model == "unet":
        print("prediction model: UNet3d")
        pred_model = UNet3d(in_channels=3, out_channels=3, time_dim=VIDEO_IN_LENGTH).to(DEVICE)
    elif cfg.model == "lstm":
        print("prediction model: LSTM")
        pred_model = LSTMModel(in_channels=3, out_channels=3).to(DEVICE)
    else:
        print("prediction model: CopyLastFrame")
        raise NotImplementedError  # TODO skip training
        pred_model = CopyLastFrameModel().to(DEVICE)

    # LOSSES AND METRICS
    concat_input_for_loss = False
    if cfg.loss == "fvd":
        print("loss function: FVD")
        loss_fn = FrechetVideoDistance(num_frames=VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH, in_channels=3)
        concat_input_for_loss = True
    else:
        print("loss function: MSE")
        loss_fn = torch.nn.MSELoss()

    # OPTIMIZER
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
            data = data.to(DEVICE)  # [b, T, c, h, w], with T = in_length + pred_length
            input = data[:, :VIDEO_IN_LENGTH]
            targets = data if concat_input_for_loss else data[:, VIDEO_IN_LENGTH:]
            predictions = pred_model.pred_n(input, pred_length=VIDEO_PRED_LENGTH)
            if concat_input_for_loss:
                predictions = torch.cat([input, predictions], dim=1)
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        print("Validating...")
        cur_loss = validate_video_model(valid_loader, pred_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, loss_fn, concat_input_for_loss)
        print("Val. loss = {}".format(cur_loss))

        # save model if last epoch improved acc.
        if min_loss > cur_loss:
            max_accuracy = cur_loss
            torch.save(pred_model, str((out_dir / 'best_model.pth').resolve()))
            print('Model saved!')

        # visualize model predictions using eval mode and validation data
        print("Saving visualizations...")
        visualize_video(val_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, pred_model, out_dir, num_vis=10)

        if i == 25:
            optimizer.param_groups[0]['lr'] *= 0.1
            print('Decrease learning rate!')

    # TESTING
    print("\nTraining done, testing best model...")
    best_model = torch.load(str((out_dir / 'best_model.pth').resolve()))
    test_data = SynpickVideoDataset(data_dir=test_dir, num_frames=VIDEO_IN_LENGTH + VIDEO_PRED_LENGTH)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    test_loss = validate_video_model(test_loader, best_model, DEVICE, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, loss_fn, concat_input_for_loss)
    print("Test MSE loss = {}".format(test_loss))

    print("Saving visualizations...")
    visualize_video(test_data, VIDEO_IN_LENGTH, VIDEO_PRED_LENGTH, best_model, out_dir, num_vis=10)
    print("Testing done, bye bye!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Video Prediction Model Training")
    parser.add_argument("--in-path", type=str, help="Path to dataset directory")
    parser.add_argument("--model", type=str, choices=["unet", "lstm"], help="Which model arch to train")
    parser.add_argument("--loss", type=str, choices=["mse", "fvd"], help="Which loss function to use for training")

    cfg = parser.parse_args()
    main(cfg)