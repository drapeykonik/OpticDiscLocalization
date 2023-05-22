import argparse

from exp_runner.runner import evaluate

parser = argparse.ArgumentParser(description="Runner")
parser.add_argument("model")
parser.add_argument("experiment")
parser.add_argument("image")
args = vars(parser.parse_args())
evaluate(args["model"], args["experiment"], args["image"])
