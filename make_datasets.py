import os
import random

from labels import labels


PATH = '~/Desktop/101_ObjectCategories/'
SPLIT = 0.8 # Train

train_data = []
test_data = []


for root, dirs, files in os.walk(os.path.expanduser(PATH)):
    label = os.path.basename(root)

    # Skip non matching labels.
    if label not in labels:
        continue

    for f in files:
        abspath = os.path.join(root, f)
        is_train = random.random() < SPLIT
        if is_train:
            train_data.append((abspath, label))
        else:
            test_data.append((abspath, label))

random.shuffle(train_data)
random.shuffle(test_data)

with open('train_data.txt', 'w') as f:
    for path, label in train_data:
        f.write('%s %s\n' % (path, label))

with open('test_data.txt', 'w') as f:
    for path, label in test_data:
        f.write('%s %s\n' % (path, label))

print len(train_data), len(test_data)

