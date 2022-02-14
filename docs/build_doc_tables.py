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
    model_info_header = ["Model Name", "Model Identifier", "Paper Reference", "Code Reference", "Matches Reference?"]
    model_info_list = list()
    for model_id, model_class in MODEL_CLASSES.items():
        cur_model_info = [model_class.NAME, f"``{model_id}``", model_class.PAPER_REFERENCE or "",
                          model_class.CODE_REFERENCE or "", model_class.MATCHES_REFERENCE or ""]
        model_info_list.append(cur_model_info)
    _build_table(model_info_list, model_info_header, "Available Models", "source/available_models.rst")


def build_available_datasets_table():
    pass


def build_available_model_blocks_table():
    pass


def build_available_metrics_table():
    pass


def build_available_losses_table():
    pass


if __name__ == '__main__':
    build_available_models_table()
    build_available_losses_table()
    build_available_metrics_table()
    build_available_model_blocks_table()
    build_available_datasets_table()
