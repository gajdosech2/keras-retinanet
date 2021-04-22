#!/usr/bin/env python
# coding: utf-8

# Load necessary modules
import sys
sys.path.insert(0, '../')

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import cv2
import os
import numpy as np
import time
from PIL import Image

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


RESULT_PATH = "result/"
PROCESS_PATH = "process/"
DATA_SUFFIX = '_datamap.png'

model_path = os.path.join('snapshots/', '', 'model.h5')
model = models.load_model(model_path, backbone_name='resnet50')
print(model.summary())

for f in os.listdir(PROCESS_PATH):
    if f.endswith(DATA_SUFFIX):
        image = read_image_bgr(PROCESS_PATH + f)
        draw = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        
        print(image.shape)
        
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print('processing time: ', time.time() - start)
        boxes /= scale
        
        labels_to_names = {0:'plate', 1:'thruster', 2:'circlet'}
        
        with open(RESULT_PATH + f[:-len(DATA_SUFFIX)] + '.txt', 'w') as output:
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score < 0.8:
                    break

                color = label_color(label)
                b = box.astype(int)
                print(','.join(map(str, list(b))) + ',' + str(label), file=output)
                draw_box(draw, b, color=[(0, 255, 0), (255, 0, 0), (0, 0, 255)][label])

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)

            im = Image.fromarray(draw)
            im.save(RESULT_PATH + f[:-len(DATA_SUFFIX)] + '_result.png')
   

