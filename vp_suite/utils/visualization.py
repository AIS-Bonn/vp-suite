import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import cv2 as cv
except ImportError:
    raise ImportError("importing cv2 failed -> please install opencv-python (cv2) "
                      "or use the gif-mode for visualization.")

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


def add_borders(trajs, context_frames):
    T, h, w, _ = list(trajs.values())[0].shape
    border_width = min(h, w) // 8
    for key, traj in trajs.items():
        if "true_" in key.lower() or "gt_" in key.lower() or key.lower() == "gt":
            trajs[key] = add_border_around_vid(traj, [("green", T)], b_width=border_width)
        elif "seg" in key.lower():
            trajs[key] = add_border_around_vid(traj, [("yellow", T)], b_width=border_width)
        else:
            trajs[key] = add_border_around_vid(traj, [("green", context_frames), ("red", T - context_frames)],
                                               b_width=border_width)
    return trajs


def save_vid_vis(out_fp, context_frames, mode="gif", **trajs):
    trajs = {k: v for k, v in trajs.items() if v is not None}  # filter out 'None' trajs
    T, h, w, _ = list(trajs.values())[0].shape
    trajs = add_borders(trajs, context_frames)

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

        combined_traj = np.concatenate(list(trajs.values()), axis=-2)  # put visualizations next to each other
        out_paths = []
        for t, frame in enumerate(list(combined_traj)):
            out_fn = f"{out_fp[:-4]}_t{t}.jpg"
            out_paths.append(out_fn)
            out_frame_BGR = frame[:, :, ::-1]
            cv.imwrite(out_fn, out_frame_BGR)
        clip = ImageSequenceClip(out_paths, fps=2)
        clip.write_videofile(f"{out_fp[:-4]}.mp4", fps=2)
        for out_fn in out_paths:
            os.remove(out_fn)


def get_vis_from_model(dataset, data, pred_model, data_unpack_config, pred_frames):
    pred_model.eval()

    # data prep
    if pred_model.NEEDS_COMPLETE_INPUT:
        input, _, actions = pred_model.unpack_data(data, data_unpack_config)
        input_vis = dataset.postprocess(input.clone().squeeze(dim=0))
    else:
        input, target, actions = pred_model.unpack_data(data, data_unpack_config)
        full = torch.cat([input.clone(), target.clone()], dim=1)
        input_vis = dataset.postprocess(full.squeeze(dim=0))

    # fwd
    with torch.no_grad():
        pred, _ = pred_model(input, pred_frames, actions=actions)  # [1, T_pred, c, h, w]

    # assemble prediction
    if pred_model.NEEDS_COMPLETE_INPUT:  # replace original pred frames with actual prediction
        input_and_pred = input
        input_and_pred[:, -pred.shape[1]:] = pred
    else:  # concat context frames and prediction
        input_and_pred = torch.cat([input, pred], dim=1)  # [1, T, c, h, w]
    pred_vis = dataset.postprocess(input_and_pred.squeeze(dim=0))  # [T, h, w, c]

    pred_model.train()
    return input_vis, pred_vis


def visualize_vid(dataset, context_frames, pred_frames, pred_model, device,
                  out_path, vis_idx, vis_mode):

    out_fn_template = "vis_{}." + vis_mode
    data_unpack_config = {"device": device, "context_frames": context_frames, "pred_frames": pred_frames}

    if vis_idx is None or any([x >= len(dataset) for x in vis_idx]):
        raise ValueError(f"invalid vis_idx provided for visualization "
                         f"(dataset len = {len(dataset)}): {vis_idx}")

    for i, n in enumerate(vis_idx):
        # prepare input and ground truth sequence
        input_vis, pred_vis = get_vis_from_model(dataset, dataset[n], pred_model,
                                                 data_unpack_config, pred_frames)
        # visualize
        out_filename = str(out_path / out_fn_template.format(str(i)))
        save_vid_vis(out_fp=out_filename, context_frames=context_frames,
                     GT=input_vis, Pred=pred_vis, mode=vis_mode)


def save_frame_compare_vis(out_filename, T_context, ground_truth_vis,
                           preds_vis, vis_context_frame_idx):
    border = 2
    all_seqs = [ground_truth_vis] + preds_vis
    T, h, w, c = ground_truth_vis.shape
    hb, wb = h + border, w + border  # img sizes with borders
    H = (hb * len(all_seqs)) - border  # height of resulting vis.
    W_context = (wb * len(vis_context_frame_idx))  # width of context part of resulting vis.
    W_pred = (wb * (T - T_context))  # width of prediction part of resulting vis.

    # left part of seq vis: only first row is populated (with context frames)
    large_img_context = np.ones((H, W_context, c), dtype=np.uint8) * 255
    for n_frame, context_i in enumerate(vis_context_frame_idx):
        w_start = n_frame * wb
        large_img_context[:h, w_start:w_start+w] = ground_truth_vis[context_i]

    # right part of seq vis: display predictions below ground truth frame-by-frame
    large_img_pred = np.ones((H, W_pred, c), dtype=np.uint8) * 255
    for n_seq, seq in enumerate(all_seqs):
        for t in range(T_context, T):
            h_start = n_seq * hb
            w_start = (t - T_context) * wb + border
            large_img_pred[h_start:h_start+h, w_start:w_start+w, :] = seq[t]

    large_img = np.concatenate([large_img_context, large_img_pred], axis=-2)
    cv.imwrite(out_filename, cv.cvtColor(large_img, cv.COLOR_RGB2BGR))


def visualize_sequences(dataset, context_frames, pred_frames, models, device,
                        out_path, vis_idx, vis_context_frame_idx):

    vis_out_fn_template = "vis_{}.png"
    data_unpack_config = {"device": device, "context_frames": context_frames, "pred_frames": pred_frames}
    info_file_lines = [f"DATASET: {dataset.NAME}", f"chosen dataset idx: {vis_idx}",
                       f"Displayed context frames: {vis_context_frame_idx}",
                       f"Displayed pred frames: {list(range(context_frames, context_frames+pred_frames))}",
                       "Displayed rows (from top):", " - Ground Truth"]

    if vis_idx is None or any([x >= len(dataset) for x in vis_idx]):
        raise ValueError(f"invalid vis_idx provided for visualization "
                         f"(dataset len = {len(dataset)}): {vis_idx}")

    for i, n in enumerate(vis_idx):
        data = dataset[n]  # [T, c, h, w]
        ground_truth_vis = None
        preds_vis = []
        for pred_model in models:
            if pred_model.model_dir is None:  # skip baseline models such as CopyLstFrame
                continue

            input_vis, pred_vis = get_vis_from_model(dataset, data, pred_model,
                                                     data_unpack_config, pred_frames)
            if ground_truth_vis is None:
                ground_truth_vis = input_vis
            preds_vis.append(pred_vis)

            if i == 0:
                info_file_lines.append(f" - {pred_model.NAME} (model dir: {pred_model.model_dir})")

        # visualize
        vis_out_fn = str((out_path / vis_out_fn_template.format(str(i))).resolve())
        save_frame_compare_vis(vis_out_fn, context_frames, ground_truth_vis,
                               preds_vis, vis_context_frame_idx)

        info_file_lines.append(f"vis {i} (idx {n}) origin: {data['origin']}")

    vis_info_fn = str((out_path / "vis_info.txt").resolve())
    with open(vis_info_fn, "w") as vis_info_file:
        vis_info_file.writelines(line + '\n' for line in info_file_lines)


def save_diff_hist(diff, diff_id):
    avg_diff, min_diff, max_diff = np.average(diff), np.min(diff), np.max(diff)
    plt.hist(diff.flatten(), bins=1000, log=True)
    plt.suptitle(f"np.abs(their_pred - our_pred)\n"
                 f"min: {min_diff}, max: {max_diff}, avg: {avg_diff}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(f"diff_{diff_id}.png")
