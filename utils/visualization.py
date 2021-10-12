import math
import os

import networkx as nx
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import to_networkx

from dataset.graph.synpick_graph import denormalize_t, tote_min_coord, tote_max_coord
from dataset.dataset_utils import postprocess_img, postprocess_mask, colorize_semseg
from utils.quaternion import dq_translation, dq_normalize
from pyquaternion import Quaternion

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


# === VID_SEG ==================================================================

def save_seg_vis(out_fp, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(out_fp)


def visualize_seg(dataset, seg_model, device, out_dir=".", num_vis=5):

    for i in range(num_vis):
        n = np.random.choice(len(dataset))

        image, gt_mask = dataset[n]
        image_vis = postprocess_img(image.permute((1, 2, 0)))
        gt_mask_vis = postprocess_mask(gt_mask.squeeze())
        gt_mask_vis = colorize_semseg(gt_mask_vis, num_classes=dataset.num_classes)

        if seg_model is not None:
            seg_model.eval()
            with torch.no_grad():
                pr_mask = seg_model(image.to(device).unsqueeze(0))
                pr_mask_vis = postprocess_mask(pr_mask.argmax(dim=1).squeeze())
                pr_mask_vis = colorize_semseg(pr_mask_vis, num_classes=dataset.num_classes)

                save_seg_vis(
                    out_fp=os.path.join(out_dir, "{}.png".format(str(i))),
                    image=image_vis,
                    ground_truth_mask=gt_mask_vis,
                    predicted_mask=pr_mask_vis
                )
            seg_model.train()

        else:
            save_seg_vis(
                out_fp=os.path.join(out_dir, "{}.png".format(str(i))),
                image=image_vis,
                ground_truth_mask=gt_mask_vis
            )


# === VID_PRED =================================================================

def get_color_array(color):
    if color == "green":
        color_array = np.array([0, 200, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    elif color == "red":
        color_array = np.array([150, 0, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    elif color == "yellow":
        color_array = np.array([100, 100, 0], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    else:
        color_array = np.array([255, 255, 255], dtype=np.uint8)[np.newaxis, ..., np.newaxis, np.newaxis]
    return color_array


def add_border_around_vid(vid, c_and_l, b_width=10):

    _, _, h, w = vid.shape
    color_bars_vertical = [np.tile(get_color_array(c), (l, 1, h, b_width)) for (c, l) in c_and_l]
    cbv = np.concatenate(color_bars_vertical, axis=0)

    color_bars_horizontal = [np.tile(get_color_array(c), (l, 1, b_width, w + 2 * b_width)) for (c, l) in c_and_l]
    cbh = np.concatenate(color_bars_horizontal, axis=0)

    vid = np.concatenate([cbv, vid, cbv], axis=-1)   # add bars in the width dim
    vid = np.concatenate([cbh, vid, cbh], axis=-2)   # add bars in the height dim
    return vid


def save_vid_vis(out_fp, vid_input_length, **trajs):

    T, _, h, w = list(trajs.values())[0].shape
    T_in, T_pred = vid_input_length, T-vid_input_length
    for key, traj in trajs.items():
        if "true_" in key.lower() or "gt_" in key.lower():
            trajs[key] = add_border_around_vid(traj, [("green", T)], b_width=16)
        elif "seg" in key.lower():
            trajs[key] = add_border_around_vid(traj, [("yellow", T)], b_width=16)
        else:
            trajs[key] = add_border_around_vid(traj, [("green", T_in), ("red", T_pred)], b_width=16)

    n_trajs = len(trajs)
    plt_scale = 0.01
    plt_cols = math.ceil(math.sqrt(n_trajs))
    plt_rows = math.ceil(n_trajs / plt_cols)
    plt_w = 1.2 * w * plt_scale * plt_cols
    plt_h = 1.4 * h * plt_scale * plt_rows
    fig = plt.figure(figsize=(plt_w, plt_h), dpi=100)

    def update(t):
        for i, (name, traj) in enumerate(trajs.items()):
            plt.subplot(plt_rows, plt_cols, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(traj[t].transpose(1, 2, 0))

    anim = FuncAnimation(fig, update, frames=np.arange(T), interval=500)
    anim.save(out_fp, writer="imagemagick", dpi=200)
    plt.close(fig)


def visualize_vid(dataset, vid_input_length, vid_pred_length, pred_model, device,
                  out_dir=".", vid_type=("rgb", 3), num_vis=5, test=False):

    pred_mode, num_channels = vid_type
    out_fn_template = "vis_{}_test.gif" if test else "vis_{}.gif"
    out_filenames = []

    for i in range(num_vis):

        out_filename = os.path.join(out_dir, out_fn_template.format(str(i)))
        out_filenames.append(out_filename)
        n = np.random.choice(len(dataset))
        data = dataset[n] # [in_l + pred_l, c, h, w]

        gt_rgb_vis = postprocess_img(data["rgb"])
        gt_colorized_vis = postprocess_img(data["colorized"])
        actions = data["actions"].to(device).unsqueeze(dim=0)
        in_traj = data[pred_mode]

        if pred_model is not None:
            pred_model.eval()
            with torch.no_grad():
                in_traj = in_traj[:vid_input_length].to(device).unsqueeze(dim=0)  # [1, in_l, c, h, w]
                pr_traj, _ = pred_model.pred_n(in_traj, vid_pred_length, actions=actions)  # [1, pred_l, c, h, w]
                pr_traj = torch.cat([in_traj, pr_traj], dim=1) # [1, in_l + pred_l, c, h, w]

                if num_channels == 3:
                    pr_traj_vis = postprocess_img(pr_traj.squeeze(dim=0))  # [in_l + pred_l, c, h, w]
                else:
                    pr_traj_vis = postprocess_mask(pr_traj.argmax(dim=2).squeeze())  # [in_l + pred_l, h, w]
                    pr_traj_vis = colorize_semseg(pr_traj_vis, num_classes=num_channels).transpose((0, 3, 1, 2))  # [in_l + pred_l, 3, h, w]

                save_vid_vis(out_fp=out_filename, vid_input_length=vid_input_length, true_trajectory=gt_rgb_vis,
                    true_colorized=gt_colorized_vis, pred_trajectory=pr_traj_vis)

            pred_model.train()

        else:
            save_vid_vis(out_fp=out_filename, vid_input_length=vid_input_length, true_trajectory=gt_rgb_vis,
                true_colorized=gt_colorized_vis)

    return out_filenames


# === GRAPH ====================================================================

ROT_AX_LEN = 16
NODE_COLORMAP = plt.cm.get_cmap("gist_ncar")
MAX_CLASS_ID = 22

def network_plot_3D(ax, G, graph_mode, alpha, angle=0, out_fp=None):

    node_xs = nx.get_node_attributes(G, 'x')
    node_colors = [NODE_COLORMAP(node_x[-1] / MAX_CLASS_ID) for _, node_x in node_xs.items()]
    edge_alphas = [min(w, 1.0) for _, _, w in G.edges.data("edge_attr")]

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for key, node_x in node_xs.items():

        # position visualization
        t_vec = node_x[4:-1] if graph_mode == "t" else dq_translation(dq_normalize(torch.tensor(node_x[:-1])))
        x, y, z = denormalize_t(np.array(t_vec))
        node_m = "s" if node_x[-1] == MAX_CLASS_ID else "o"
        node_ec = "k" if node_x[-1] == MAX_CLASS_ID else "face"
        ax.scatter(x, y, z, s=100, color=node_colors[key], marker=node_m, edgecolors=node_ec, alpha=alpha)

        # rotation visualization
        if graph_mode == "dq":
            R = Quaternion(node_x[:4]).rotation_matrix * ROT_AX_LEN  # scale by ROT_AX_LEN for vis. purposes
            ax.plot(np.array((x, x + R[0, 0])), np.array((y, y + R[1, 0])), np.array((z, z + R[2, 0])),
                    c="r", alpha=alpha, linewidth=2)  # rot_ax_x
            ax.plot(np.array((x, x + R[0, 1])), np.array((y, y + R[1, 1])), np.array((z, z + R[2, 1])),
                    c="g", alpha=alpha, linewidth=2)  # rot_ax_y
            ax.plot(np.array((x, x + R[0, 2])), np.array((y, y + R[1, 2])), np.array((z, z + R[2, 2])),
                    c="b", alpha=alpha, linewidth=2)  # rot_ax_z

    # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
    # Those two points are the extrema of the line to be plotted
    for i, (e_start, e_end) in enumerate(G.edges()):
        if graph_mode == "t":
            t1_vec = node_xs[e_start][4:-1]
            t2_vec = node_xs[e_end][4:-1]
        else:  # graph_mode == "dq"
            t1_vec = dq_translation(dq_normalize(torch.tensor(node_xs[e_start][:-1])))
            t2_vec = dq_translation(dq_normalize(torch.tensor(node_xs[e_end][:-1])))

        # Plot the connecting lines
        x1, y1, z1 = denormalize_t(np.array(t1_vec))
        x2, y2, z2 = denormalize_t(np.array(t2_vec))
        ax.plot(np.array((x1, x2)), np.array((y1, y2)), np.array((z1, z2)),
                c='black', alpha=edge_alphas[i]*alpha, linewidth=0.5)

    # Set the initial view
    ax.view_init(20, angle)
    ax.set_xlim(tote_min_coord[0], tote_max_coord[0]);
    ax.set_ylim(tote_min_coord[1], tote_max_coord[1]);
    ax.set_zlim(1800, tote_max_coord[2]);

    if out_fp is not None:
        plt.savefig(out_fp)
        plt.close('all')

    return


def draw_synpick_pred_and_gt(graph_pred, graph_target, graph_mode, out_fp, frame=0):

    G_pred = to_networkx(graph_pred, to_undirected=True, node_attrs=["x"], edge_attrs=["edge_attr"])
    G_target = to_networkx(graph_target, to_undirected=True, node_attrs=["x"], edge_attrs=["edge_attr"])

    # create plot and save
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    network_plot_3D(ax, G_pred, graph_mode, alpha=1.0, angle=1*frame)
    network_plot_3D(ax, G_target, graph_mode, alpha=0.25, angle=1*frame, out_fp=out_fp)

def visualize_graph(cfg, vis_pairs, test=False):

    from moviepy.editor import ImageSequenceClip

    out_fname_template = "vis_{}_test.gif" if test else "vis_{}.gif"
    out_fname_g_template = "vis_{}_t{}_test.png" if test else "vis_{}_t{}.png"
    out_filenames = []

    for g, (signal_pred, signal_in) in enumerate(vis_pairs):
        out_g_filenames = []
        for t, (snap_pred, snap_target) in enumerate(zip(signal_pred, signal_in)):
            out_g_fname = os.path.join(cfg.out_dir, out_fname_g_template.format(g, t))
            out_g_filenames.append(out_g_fname)
            draw_synpick_pred_and_gt(snap_pred, snap_target, cfg.graph_mode, out_g_fname, frame=t)

        clip = ImageSequenceClip(out_g_filenames, fps=2)
        out_fname = os.path.join(cfg.out_dir, out_fname_template.format(g))
        out_filenames.append(out_fname)
        clip.write_gif(out_fname, fps=2)
        for out_g_fname in out_g_filenames:
            os.remove(out_g_fname)

    return out_filenames