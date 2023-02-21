import os

path = '/mnt/datasets/class_1/1.jpg'
gt = os.path.dirname(path).split(os.path.sep)[-1]
print(gt)