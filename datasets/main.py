import json
import argparse
import time

from name import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=AMAZON, help='Dataset used')
parser.add_argument('--seed', type=int, default=111, help='seed')
args = parser.parse_args()

print("Model info:")
print(json.dumps(args.__dict__, indent='\t'))

data = args.data
seed = args.seed

utils.set_seed(seed)


print("Generating data...")
s = time.time()
y = utils.load_data(data)
utils.split_train_val_test(data, y)
e = time.time()
print("Generating successfully! Time cost: {}".format(e - s))
