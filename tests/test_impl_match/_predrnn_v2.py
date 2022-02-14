import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from vp_suite.models.predrnn_v2 import PredRNN_V2
from vp_suite.utils.models import state_dicts_equal

REFERENCE_GIT_URL = "https://github.com/thuml/predrnn-pytorch.git"
REPO_DIR = "predrnn-pytorch"

def test_impl():

    parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

    # training/test
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu:0')

    # data
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
    parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
    parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
    parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--total_length', type=int, default=20)
    parser.add_argument('--img_width', type=int, default=64)
    parser.add_argument('--img_channel', type=int, default=1)

    # model
    parser.add_argument('--model_name', type=str, default='predrnn')
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--decouple_beta', type=float, default=0.1)

    # reverse scheduled sampling
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
    parser.add_argument('--r_sampling_step_1', type=int, default=25000)
    parser.add_argument('--r_sampling_step_2', type=int, default=50000)
    parser.add_argument('--r_exp_alpha', type=int, default=5000)
    # scheduled sampling
    parser.add_argument('--scheduled_sampling', type=int, default=1)
    parser.add_argument('--sampling_stop_iter', type=int, default=50000)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

    # optimization
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reverse_input', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_iterations', type=int, default=80000)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=5000)
    parser.add_argument('--snapshot_interval', type=int, default=5000)
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--n_gpu', type=int, default=1)

    # visualization of memory decoupling
    parser.add_argument('--visual', type=int, default=0)
    parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

    # action-based predrnn
    parser.add_argument('--injection_action', type=str, default='concat')
    parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
    parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
    parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')
    args = parser.parse_args([])  # this is called from a pytest test, so parse from empty list

    args.is_training = 0
    args.device = "cuda"
    args.img_channel = 3
    args.batch_size = 1
    args.num_hidden = "128, 128, 128, 128"

    # test different architectures for equality
    action_setups = [
        {"model_name": "predrnn_v2", "reverse_scheduled_sampling": 0},
        {"model_name": "predrnn_v2", "reverse_scheduled_sampling": 1},
        # the original action-conditonal implementations are broken: if using the action-conditional variant,
        # reverse scheduled sampling as well as 'conv_on_input' has to be set to 1/True!
        {"model_name": "action_cond_predrnn_v2", "conv_on_input": 1, "reverse_scheduled_sampling": 1, "res_on_conv": 0},
        {"model_name": "action_cond_predrnn_v2", "conv_on_input": 1, "reverse_scheduled_sampling": 1, "res_on_conv": 1}
    ]
    test_setups = []
    for action_setup in action_setups:
        for layer_norm in [0, 1]:
            test_setup = deepcopy(action_setup)
            test_setup["layer_norm"] = layer_norm
            test_setups.append(test_setup)

    for i, test_setup in enumerate(test_setups):
        print(f"\ntest #{i}, setup: {test_setup}")
        cur_args = deepcopy(args)
        for key, val in test_setup.items():
            setattr(cur_args, key, val)
        test_predrnnv2(cur_args, i)


def test_predrnnv2(args, test_id):
    from core.models.model_factory import Model
    from core.utils.preprocess import reshape_patch, reshape_patch_back

    device = args.device
    # set up original model
    print("setting up their model")
    their_model: nn.Module = Model(args).network.to(device)

    # set up our model
    print("setting up our model")
    args_dict = _prepare_args_for_our_model(deepcopy(args))
    our_model: nn.Module = PredRNN_V2(device, **args_dict).to(device)

    # check and assign state dicts
    print("checking model state dicts")
    assert state_dicts_equal(their_model, our_model), "State dicts not equal!"
    our_model.load_state_dict(deepcopy(their_model.state_dict()))
    assert state_dicts_equal(their_model, our_model, check_values=True), "State dicts not equal!"

    # set up input
    print("setting up input")
    t_input = args.input_length
    t_total = args.total_length
    t_pred = t_total - t_input
    c = args.img_channel
    h = w = args.img_width
    a = args.num_action_ch
    their_x = np.random.random_sample((args.batch_size, t_total, h, w, c))
    our_x = np.copy(their_x)
    their_a = np.random.random_sample((args.batch_size, t_total, a))
    our_a = np.copy(their_a)

    # infer: their model
    print("infer: theirs")
    their_model.eval()
    their_x_patch = reshape_patch(their_x, args.patch_size)
    if "action_cond" in args.model_name:
        their_a_patch = np.tile(their_a[:, :, np.newaxis, np.newaxis, :],
                                (1, 1, their_model.patch_height, their_model.patch_width, 1))  # [b, t, h_, w_, cpp+a]
        their_x_patch = np.concatenate([their_x_patch, their_a_patch], axis=-1)
    mask_input = 1 if args.reverse_scheduled_sampling == 1 else args.input_length
    their_mask_patch = np.zeros((args.batch_size, args.total_length - mask_input - 1,
                                 args.img_width // args.patch_size,
                                 args.img_width // args.patch_size,
                                 args.patch_size ** 2 * args.img_channel))
    if args.reverse_scheduled_sampling == 1:
        their_mask_patch[:, :args.input_length - 1, :, :] = 1.0
    their_x_patch = torch.FloatTensor(their_x_patch).to(device)
    their_mask_patch = torch.FloatTensor(their_mask_patch).to(device)
    with torch.no_grad():
        their_result = their_model(their_x_patch, their_mask_patch)
    their_pred_patch = their_result[0]
    their_pred = reshape_patch_back(their_pred_patch.cpu().numpy(), args.patch_size)
    output_length = args.total_length - args.input_length
    their_pred = their_pred[:, -output_length:]
#
    # infer: our model
    print("infer: ours")
    our_model.eval()
    our_x = torch.FloatTensor(our_x).permute((0, 1, 4, 2, 3)).to(device)
    our_a = torch.FloatTensor(our_a).to(device)
    with torch.no_grad():
        our_result = our_model(our_x, pred_frames=t_pred, actions=our_a)
    our_pred = our_result[0]
    our_pred = our_pred.permute((0, 1, 3, 4, 2)).cpu().numpy()

    # checks
    print("check results")
    assert their_pred.shape == our_pred.shape, f"Prediction shapes are not equal. " \
                                               f"Theirs: {their_pred.shape}, ours: {our_pred.shape}"
    # save_diff_hist(np.abs(their_pred - our_pred), test_id)
    assert np.allclose(their_pred, our_pred, rtol=0, atol=1e-4), "Predictions are not equal."


def _prepare_args_for_our_model(args_namespace):
    args_dict = vars(args_namespace)

    # to bool
    keys_to_bool = ["layer_norm", "reverse_scheduled_sampling", "scheduled_sampling",
                    "reverse_input", "conv_on_input", "res_on_conv"]
    for key in keys_to_bool:
        args_dict[key] = True if args_dict[key] == 1 else False

    # renames, prep etc.
    args_dict["action_size"] = args_dict["num_action_ch"]
    args_dict["context_frames"] = args_dict["input_length"]
    args_dict["total_frames"] = args_dict["total_length"]
    args_dict["action_conditional"] = "action_cond" in args_dict["model_name"]
    args_dict["conv_actions_on_input"] = args_dict["conv_on_input"]
    args_dict["residual_on_action_conv"] = args_dict["res_on_conv"]
    args_dict["img_shape"] = (args_dict["img_channel"], args_dict["img_width"], args_dict["img_width"])
    args_dict["pred_frames"] = args_dict["total_frames"] - args_dict["context_frames"] - 1
    args_dict["tensor_value_range"] = [0.0, 1.0]
    args_dict["num_hidden"] = [int(x) for x in args_dict["num_hidden"].split(',')]
    args_dict["num_layers"] = len(args_dict["num_hidden"])

    # remove keys
    keys_to_remove = ["is_training", "dataset_name", "train_data_paths", "valid_data_paths", "save_dir", "gen_frm_dir",
                      "img_width", "img_channel", "model_name", "pretrained_model", "lr", "max_iterations",
                      "display_interval", "test_interval", "snapshot_interval", "num_save_samples", "n_gpu", "visual",
                      "visual_path", "injection_action", "num_action_ch", "device"]
    for key in keys_to_remove:
        args_dict.pop(key, None)
    return args_dict
