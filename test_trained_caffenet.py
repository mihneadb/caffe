import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil

import caffe

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])


from labels import labels
ALLOWED_LABELS = []
for v in labels.values():
    ALLOWED_LABELS.extend(v)
ALLOWED_LABELS = set(ALLOWED_LABELS)


def classify(filename):
    image = exifutil.open_oriented_im(filename)
    result = clf.classify_image(image)
    return result


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/caffenet_train_iter_100000.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
             channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            return indices[0]

            # Keep meaningful labels.
            relevant_indices = []
            for i in indices:
                if i in ALLOWED_LABELS:
                    relevant_indices.append(i)

            if relevant_indices:
                return relevant_indices[0]
            return None


        except Exception as err:
            logging.info('Classification error: %s', err)
            return None


ImagenetClassifier.default_args.update({'gpu_mode': True})
# Initialize classifier + warm start by forward for allocation
clf = ImagenetClassifier(**ImagenetClassifier.default_args)
clf.net.forward()



logging.getLogger().setLevel(logging.INFO)


#path = '/home/mihnea/Desktop/101_ObjectCategories/wild_cat/image_0027.jpg'
#r = classify(path)
#print get_label_name(r)

good = 0
no_label = 0
total = 0.0
test_data_file = '/home/mihnea/localCode/caffe/test_data_caffe.txt'
with open(test_data_file) as f:
    for line in f:
        total += 1

        img_path, expected_label = line.split()
        r = classify(img_path)
        #print expected_label, r

        if r is None:
            no_label += 1
            continue

        if int(r) == int(expected_label):
            good += 1

print 'Accuracy, Good, No Label, Total'
print good / total * 100, good, no_label, total

