from trainer.trainer import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser("A module for creating text neural networks for music reviews")
    parser.add_argument("-c", "--config", help="The config yaml to train the model", type=str, dest="config")
    parser.add_argument("-t", "--train", help="Are we training?  If not we'll be predicting", type=bool, default=True,
                        dest="train")
    parser.add_argument("--test", help="Is this being run for testing", type=bool, default=False, dest="test")
    args = parser.parse_args()
    if args.training:
        t = Trainer(args.config, args.test)
        t.train_model()


if __name__ == '__main__':
    main()
