import math
import os

import numpy as np
import torch

COLORS = {
    "green": [0, 200, 0],
    "red": [150, 0, 0],
    "yellow": [100, 100, 0],
    "black": [0, 0, 0],
    "white": [255, 255, 255]
}  # r, g, b

def get_color_array(color):
    r_g_b = COLORS.get(color, COLORS["white"])
    return np.array(r_g_b, dtype=np.uint8)[np.newaxis, np.newaxis, np.newaxis, ...]

def add_border_around_vid(vid, c_and_l, b_width=10):

    _, h, w, _ = vid.shape
    color_bars_vertical = [np.tile(get_color_array(c), (l, h, b_width, 1)) for (c, l) in c_and_l]
    cbv = np.concatenate(color_bars_vertical, axis=0)

    color_bars_horizontal = [np.tile(get_color_array(c), (l, b_width, w + 2 * b_width, 1)) for (c, l) in c_and_l]
    cbh = np.concatenate(color_bars_horizontal, axis=0)

    vid = np.concatenate([cbv, vid, cbv], axis=-2)   # add bars in the width dim
    vid = np.concatenate([cbh, vid, cbh], axis=-3)   # add bars in the height dim
    return vid

def save_vid_vis(out_fp, context_frames, mode="gif", **trajs):

    trajs = {k: v for k, v in trajs.items() if v is not None}  # filter out 'None' trajs
    T, h, w, _ = list(trajs.values())[0].shape
    T_in, T_pred = context_frames, T-context_frames
    for key, traj in trajs.items():
        if "true_" in key.lower() or "gt_" in key.lower() or key.lower() == "gt":
            trajs[key] = add_border_around_vid(traj, [("green", T)], b_width=16)
        elif "seg" in key.lower():
            trajs[key] = add_border_around_vid(traj, [("yellow", T)], b_width=16)
        else:
            trajs[key] = add_border_around_vid(traj, [("green", T_in), ("red", T_pred)], b_width=16)

    if mode == "gif":  # gif visualizations with matplotlib  # TODO fix it
        try:
            from matplotlib import pyplot as PLT
            PLT.rcParams.update({'axes.titlesize': 'small'})
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise ImportError("importing from matplotlib failed "
                              "-> please install matplotlib or use the mp4-mode for visualization.")
        n_trajs = len(trajs)
        plt_scale = 0.01
        plt_cols = math.ceil(math.sqrt(n_trajs))
        plt_rows = math.ceil(n_trajs / plt_cols)
        plt_w = 1.2 * w * plt_scale * plt_cols
        plt_h = 1.4 * h * plt_scale * plt_rows
        fig = PLT.figure(figsize=(plt_w, plt_h), dpi=100)

        def update(t):
            for i, (name, traj) in enumerate(trajs.items()):
                PLT.subplot(plt_rows, plt_cols, i + 1)
                PLT.xticks([])
                PLT.yticks([])
                PLT.title(' '.join(name.split('_')).title())
                PLT.imshow(traj[t])

        anim = FuncAnimation(fig, update, frames=np.arange(T), interval=500)
        anim.save(out_fp, writer="imagemagick", dpi=200)
        PLT.close(fig)

    else:  # mp4 visualizations with opencv and moviepy
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            raise ImportError("importing from moviepy failed"
                              " -> please install moviepy or use the gif-mode for visualization.")
        try:
            import cv2 as cv
        except ImportError:
            raise ImportError("importing cv2 failed"
                              " -> please install opencv-python (cv2) or use the gif-mode for visualization.")
        for name, traj in trajs.items():
            frames = list(traj)
            out_paths = []
            for t, frame in enumerate(frames):
                out_fn = f"{out_fp[:-4]}_{name}_t{t}.jpg"
                out_paths.append(out_fn)
                out_frame_BGR = frame[:, :, ::-1]
                cv.imwrite(out_fn, out_frame_BGR)
            clip = ImageSequenceClip(out_paths, fps=2)
            clip.write_videofile(f"{out_fp[:-4]}_{name}.mp4", fps=2)
            for out_fn in out_paths:
                os.remove(out_fn)

def visualize_vid(dataset, context_frames, pred_frames, pred_model, device,
                  out_path, img_processor, num_vis=5, vis_idx=None, mode="gif"):

    out_fn_template = "vis_{}." + mode

    if vis_idx is None:
        vis_idx = np.random.choice(len(dataset), num_vis, replace=False)

    for i, n in enumerate(vis_idx):
        out_filename = str(out_path / out_fn_template.format(str(i)))
        data = dataset[n] # [in_l + pred_l, c, h, w]

        gt_rgb_vis = img_processor.postprocess_img(data["frames"][:context_frames+pred_frames])
        gt_colorized_vis = data.get("colorized", None)
        if gt_colorized_vis is not None:
            gt_colorized_vis = img_processor.postprocess_img(gt_colorized_vis)  # [in_l, h, w, c]
        actions = data["actions"].to(device).unsqueeze(dim=0)
        in_traj = data["frames"]

        if pred_model is not None:
            pred_model.eval()
            with torch.no_grad():
                in_traj = in_traj[:context_frames].to(device).unsqueeze(dim=0)  # [1, in_l, c, h, w]
                pr_traj, _ = pred_model(in_traj, pred_frames, actions=actions)  # [1, pred_l, c, h, w]
                pr_traj = torch.cat([in_traj, pr_traj], dim=1) # [1, in_l + pred_l, c, h, w]
                pr_traj_vis = img_processor.postprocess_img(pr_traj.squeeze(dim=0))  # [in_l + pred_l, h, w, c]

                save_vid_vis(out_fp=out_filename, context_frames=context_frames, GT=gt_rgb_vis,
                    GT_Color=gt_colorized_vis, Pred=pr_traj_vis, mode=mode)

            pred_model.train()

        else:
            save_vid_vis(out_fp=out_filename, context_frames=context_frames, GT=gt_rgb_vis,
                GT_Color=gt_colorized_vis, mode=mode)
