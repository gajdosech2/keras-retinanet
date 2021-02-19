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


PROCESS_PATH = "data/process/"
RESULT_PATH = "data/result/"

model_path = os.path.join('snapshots/', '', 'model.h5')

model = models.load_model(model_path, backbone_name='resnet50')
print(model.summary())

# load image
image = read_image_bgr(PROCESS_PATH + os.listdir(PROCESS_PATH)[0])

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# load label to names mapping for visualization purposes
labels_to_names = {0: 'wheel'}

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=(0, 255, 0))

    #caption = "{} {:.3f}".format(labels_to_names[label], score)
    #draw_caption(draw, b, caption)


im = Image.fromarray(draw)
im.save(RESULT_PATH + "result.png")
