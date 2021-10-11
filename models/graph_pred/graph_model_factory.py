from rgcn import RecurrentGCN
from rgcn_noedge import ObjectPoseEstimator


def get_graph_model(cfg, in_features):

    arch = cfg.graph_arch

    if arch == "rgcn":
        return RecurrentGCN(in_features, cfg.graph_out_features)
    elif arch == "rgcn_no_edge":
        return ObjectPoseEstimator(in_features, cfg.graph_out_features)