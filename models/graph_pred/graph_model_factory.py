from models.graph_pred.rgcn import RecurrentGCN
from models.graph_pred.rgcn_noedge import ObjectPoseEstimator


def get_graph_model(cfg):

    arch = cfg.graph_arch

    if arch == "rgcn":
        return RecurrentGCN(cfg.graph_mode, cfg.graph_in_size, cfg.graph_out_size, cfg.include_actions)
    elif arch == "rgcn_no_edge":
        return ObjectPoseEstimator(cfg.graph_mode, cfg.graph_in_size, cfg.graph_out_size, cfg.include_actions)