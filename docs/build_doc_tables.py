import sys
sys.path.append("..")
from tabulate import tabulate

from vp_suite.models import MODEL_CLASSES
# from vp_suite.model_blocks import MODEL_BLOCK_CLASSES
from vp_suite.datasets import DATASET_CLASSES
from vp_suite.measure import LOSS_CLASSES, METRIC_CLASSES


def _build_table(info_list, header, title, out_filename):
    models_table = tabulate(info_list, header, tablefmt="rst")
    with open(out_filename, "w") as models_table_file:
        models_table_file.write(f"{title}\n{''.join(['=' for _ in title])}\n\n")
        models_table_file.write(models_table)


def build_available_models_table():
    info_header = ["Model Name", "Model Identifier", "Paper Reference", "Code Reference", "Matches Reference?"]
    info_list = list()
    for model_id, model_class in MODEL_CLASSES.items():
        cur_info = [model_class.NAME, f"``{model_id}``", model_class.PAPER_REFERENCE or "",
                    model_class.CODE_REFERENCE or "", model_class.MATCHES_REFERENCE or ""]
        info_list.append(cur_info)
    _build_table(info_list, info_header, "Available Models", "source/available_models.rst")


def build_available_datasets_table():
    info_header = ["Dataset Name", "Dataset Identifier", "Reference", "Downloadable?"]
    info_list = list()
    for dataset_id, dataset_class in DATASET_CLASSES.items():
        cur_info = [dataset_class.NAME, f"``{dataset_id}``", dataset_class.REFERENCE or "",
                    dataset_class.IS_DOWNLOADABLE or ""]
        info_list.append(cur_info)
    _build_table(info_list, info_header, "Available Datasets", "source/available_datasets.rst")


def build_available_model_blocks_table():
    pass


def build_available_metrics_table():
    info_header = ["Loss Name", "Loss Identifier", "Reference"]
    info_list = list()
    for loss_id, loss_class in LOSS_CLASSES.items():
        cur_info = [loss_class.NAME, f"``{loss_id}``", loss_class.REFERENCE or ""]
        info_list.append(cur_info)
    _build_table(info_list, info_header, "Available Losses", "source/available_losses.rst")


def build_available_losses_table():
    info_header = ["Metric Name", "Metric Identifier", "Reference"]
    info_list = list()
    for metric_id, metric_class in METRIC_CLASSES.items():
        cur_info = [metric_class.NAME, f"``{metric_id}``", metric_class.REFERENCE or ""]
        info_list.append(cur_info)
    _build_table(info_list, info_header, "Available Metrics", "source/available_metrics.rst")


if __name__ == '__main__':
    build_available_models_table()
    build_available_losses_table()
    build_available_metrics_table()
    build_available_model_blocks_table()
    build_available_datasets_table()
