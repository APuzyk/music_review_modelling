import logging
from trainer.trainer import Trainer
from trainer.trainer_config import TrainerConfig
import argparse
from time import time

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
time_id = str(int(time()))

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("log_" + time_id + ".log")
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(fmt)
f_handler.setFormatter(fmt)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


def main():
    parser = argparse.ArgumentParser("A module for creating text neural networks for music reviews")
    parser.add_argument("-c", "--config", help="The config yaml to train the model", type=str, dest="config")
    parser.add_argument("-t", "--train", help="Are we training?  If not we'll be predicting", type=bool, default=True,
                        dest="train")
    parser.add_argument("--test", help="Is this being run for testing", type=bool, default=False, dest="test")
    args = parser.parse_args()
    config = TrainerConfig(args.config, args.test, time_id)
    logger.info("Running music review modeling")
    if args.train:
        t = Trainer(config)
        t.train_model()


if __name__ == '__main__':
    main()
