import sys, os
sys.path.append(".")

import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
import torch
from torch.utils.data.dataset import Dataset


class KTHActionsDataset(Dataset):

    def __init__(self, data_dir, person_ids=range(1, 26)):

        self.data_ids = [fn for fn in sorted(os.listdir(data_dir))
                         if str(fn).endswith("uncomp.avi") and int(fn[6:8]) in person_ids]
        self.activities = ['boxing', 'handclapping', 'handwaving', 'walking', 'running', 'jogging']
        self.short_activities = ['walking', 'running', 'jogging']
        self.data_fps = {
            activity: [os.path.join(data_dir, vid_id) for vid_id in self.data_ids if activity in vid_id]
            for activity in self.activities
        }

        self.seq_length = 30
        self.channels = 3
        self.action_size = 0
        self.img_shape = (64, 64)  # h, w

    def __len__(self):
        return sum([len(activity_fps) for activity_fps in self.data_fps.values()])

    def get_idx_(self, i):
        # get sequence of corresponding class
        for fps in self.data_fps.values():
            cur_class_size = len(fps)
            if i < cur_class_size:  # i in current class
                return fps[i]
            else:  # i outside current class -> subtract class_size and go to next class
                i -= cur_class_size
        raise ValueError("invalid i")

    def __getitem__(self, i):

        clip = VideoFileClip(self.get_idx_(i), audio=False, target_resolution=self.img_shape,
                             resize_algorithm="bilinear")
        seq_frames = list(clip.iter_frames(dtype='uint8'))
        last_frame = seq_frames[-1]
        while len(seq_frames) < self.seq_length:
            seq_frames.append(last_frame)  # repeat last frame if necessary
        rgb = np.stack(seq_frames[:self.seq_length], axis=0).transpose((0, 3, 1, 2))  # [t, c, h, w]

        # convert entries range from [0, 255, np.uint8] to [-1, 1, torch.float32]
        rgb = torch.from_numpy(rgb).float()
        rgb = (2 * rgb / 255) - 1

        data = {
            "rgb": rgb,
            "actions": torch.zeros((rgb.shape[0], 1))  # actions should be disregarded in training logic
        }
        return data

if __name__ == '__main__':
    data_dir = sys.argv[1]
    dataset = KTHActionsDataset(data_dir, person_ids=range(1, 17))
    print(len(dataset))
    '''
    obs = [((ob.numpy().transpose((0, 2, 3, 1)) + 1) * (255 / 2)).astype('uint8') for  (ob, _) in dataset]
    for i in range(len(obs)):
        clip = ImageSequenceClip(list(obs[i]), fps=6)
        clip.write_gif(f"results/test_{i:04d}.gif", fps=6, logger=None)
    # print(len(dataset), dp.shape, dp.min(), dp.max())
    '''