import sys
sys.path.append(".")
from vp_suite.training import Trainer

def test_full_sanity():
    vp_trainer = Trainer()
    data_dir = "/home/data/datasets/video_prediction/kth_actions"
    vp_trainer.load_dataset(dataset="KTH", data_dir=data_dir)
    vp_trainer.create_model(model_type="lstm")
    vp_trainer.train(epochs=1, no_wandb=True)
    assert True  # test successful if execution reaches this line

if __name__ == '__main__':
    test_full_sanity()