import os
import random

from labels import labels, numeric_labels, get_small_label_name


PATH = '~/localCode/caffe/train_data_caffe.txt'

with open(os.path.expanduser(PATH)) as f:
    data = f.readlines()

new_data = []

for line in data:
    name, short_label = line.split()
    new_label = random.choice(labels[get_small_label_name(short_label)])
    new_data.append('%s %s\n' % (name, new_label))

with open(os.path.expanduser('~/localCode/caffe/new_train_data_caffe.txt'), 'w') as f:
    for line in new_data:
        f.write(line)

