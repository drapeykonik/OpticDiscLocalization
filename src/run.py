import argparse

from exp_runner.runner import run

parser = argparse.ArgumentParser(description="Runner")
parser.add_argument("model")
parser.add_argument("experiment")
args = vars(parser.parse_args())
run(args["model"], args["experiment"])
