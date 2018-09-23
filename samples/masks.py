import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

from os.path import abspath, basename, join, exists
from os import walk, makedirs
from PIL import Image

# Root directory of the project
ROOT_DIR = abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# %matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = join(ROOT_DIR, "images")
RESULTS_DIR = join(IMAGE_DIR, "results")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']





# Load an image from the images folder
file_names = next(walk(IMAGE_DIR))[2]
print(len(sys.argv))
img_name = basename(sys.argv[1]) if len(sys.argv) == 2 else random.choice(file_names)
print(img_name)
image = skimage.io.imread(join(IMAGE_DIR, img_name))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]

######
r = results[0]
rois = r['rois']
new_rois = np.zeros((rois.shape[0], rois.shape[1] + 1))
new_rois[:, :-1] = rois
######

print(r['masks'][300][300])
print(new_rois.shape)

#fig, captions = visualize.visualize_instances(image, new_rois, r['masks'], r['class_ids'], 
#                           class_names, r['scores'])
captions = visualize.get_captions(new_rois, r['class_ids'], class_names, r['scores'])

visible_classes = [c.split(' ')[0] for c in captions]
print(visible_classes)

# We have pixel -> class/instance mapping, 
# but we also need class/instance -> pixel mapping.
class_names_with_indices = []
occurence_dict = dict()       # helper structure to count class occurences
class_dict = dict()
for c in visible_classes:
    if c not in occurence_dict:
        occurence_dict[c] = 0
    occurence_dict[c] += 1
    class_names_with_indices.append(c + str(occurence_dict[c]))
    class_dict[c] = np.zeros((image.shape[0], image.shape[1]))                            # per-class dictionary
    class_dict[class_names_with_indices[-1]] = np.zeros((image.shape[0], image.shape[1])) # per-instance dictionary

for height in range(image.shape[0]):
    for width in range(image.shape[1]):
        class_idx = -1
        for idx, mask_element in enumerate(r['masks'][height][width]):
            if mask_element:
                class_idx = idx
        if class_idx != -1:
            class_dict[class_names_with_indices[class_idx][:-1]][height][width] = 1.0  # per-class dictionary
            class_dict[class_names_with_indices[class_idx]][height][width] = 1.0       # per-instance dictionary

MASKS_DIR = join(RESULTS_DIR, 'masks')
IMG_MASK_DIR = join(MASKS_DIR, img_name.split('.')[0])

if not exists(MASKS_DIR):
    makedirs(MASKS_DIR)

if not exists(IMG_MASK_DIR):
    makedirs(IMG_MASK_DIR)

import scipy.misc
for c in class_dict:
    scipy.misc.imsave(join(IMG_MASK_DIR, '{}.jpg'.format(c)), class_dict[c])
