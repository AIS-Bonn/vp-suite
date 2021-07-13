import sys, os, random

import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from dataset import MyDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    # MODEL
    seg_model = smp.DeepLabV3Plus(encoder_name=SMP_ENCODER, encoder_weights=SMP_ENCODER_WEIGHTS,
                                  in_channels=3, classes=2)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(SMP_ENCODER, SMP_ENCODER_WEIGHTS)

    # DATA
    data_dir = args[0]
    train_img_dir = os.path.join(data_dir, 'train', 'rgb')
    train_msk_dir = os.path.join(data_dir, 'train', 'masks')
    val_img_dir = os.path.join(data_dir, 'val', 'rgb')
    val_msk_dir = os.path.join(data_dir, 'val', 'masks')
    test_img_dir = os.path.join(data_dir, 'test', 'rgb')
    test_msk_dir = os.path.join(data_dir, 'test', 'masks')

    SYNPICK_CLASSES = ['object_{}'.format(i) for i in range(1, 22)]

    train_data = MyDataset(images_dir=train_img_dir, masks_dir=train_msk_dir, augmentation=get_training_augmentation(),
                           preprocessing=get_preprocessing(preprocessing_fn), classes=SYNPICK_CLASSES)
    val_data = MyDataset(images_dir=val_img_dir, masks_dir=val_msk_dir, augmentation=get_validation_augmentation(),
                         preprocessing=get_preprocessing(preprocessing_fn), classes=SYNPICK_CLASSES)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    optimizer = torch.optim.Adam([
        dict(params=seg_model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        seg_model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        seg_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(seg_model, './best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    print("Training done, testing...")
    best_model = torch.load('./best_model.pth')

    test_data = MyDataset(images_dir=test_img_dir, masks_dir=test_msk_dir, augmentation=get_validation_augmentation(),
                          preprocessing=get_preprocessing(preprocessing_fn), classes=SYNPICK_CLASSES)
    test_dataloader = DataLoader(test_data)

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)


if __name__ == '__main__':
    main(sys.argv[1:])