import os
import argparse

parser = argparse.ArgumentParser(description='Train args')
parser.add_argument("--part", type=str, default="can")
args = parser.parse_args()
positive = 0
negative = 0

with open(args.part + r"/list.txt", 'w') as f:
    for filename in os.listdir(os.getcwd() + r"/" + args.part):
        if filename.endswith(".jpg"):
            f.write(filename)
            if filename.endswith(".jpg"):
                if "ng" in filename:
                    f.write(' 0\n')
                    negative += 1
                else:
                    f.write(' 1\n')
                    positive += 1

print("Positive:", positive, "Negative:", negative)