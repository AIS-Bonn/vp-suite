from pathlib import Path
from torchvision.io import read_video
from tqdm import tqdm

SCENARIOS = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting',
             'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'WalkDog', 'WalkTogether', 'Walking', 'WalkingDog']
MIN_SEQ_LEN = 994  # frames in shortest sequence (6349 in longest)
FPS = 50
SKIP_FIRST_N = 25  # some of the sequence start with a tiny bit of idling

def main():
    h36m_path = Path("/home/data/datasets/video_prediction/human3.6m")
    vids = list(h36m_path.rglob("**/*.mp4"))
    vids = [v for v in vids if not "_ALL" in str(v)]
    vid_info = []
    for vid in tqdm(vids):
        cat = str(vid.stem).split(".")[0].split(" ")[0]
        #frames = read_video(str(vid.resolve()), pts_unit="sec")[0].shape[0]
        vid_info.append((vid, cat, 0))
    frame_nums = [v[2] for v in vid_info]
    print(min(frame_nums), max(frame_nums))
    vid_cats = sorted(list(set([v[1] for v in vid_info])))
    vids_dict = {cat: [v for v in vid_info if v[1] == cat] for cat in vid_cats}
    print(vid_cats)

if __name__ == '__main__':
    main()