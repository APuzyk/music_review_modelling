from trainer.trainer import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser("A module for creating text neural networks for music reviews")
    parser.add_argument("config", help="The config yaml to train the model", type=str)
    parser.add_argument("training", help="Are we training?  If not we'll be predicting", type=bool, default=True)
    parser.add_argument("test", help="Is this being run for testing", type=bool, default=False)
    args = parser.parse_args()
    if args.training:
        t = Trainer(args.config, args.is_test)
        t.train_model()
