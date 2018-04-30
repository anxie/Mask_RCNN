import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

from mrcnn import utils
import mrcnn.model as modellib
import skimage.io
from natsort import natsorted
import numpy as np
import pickle

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGE_DIR = "/Users/anniexie/CS280_Project/images/"

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load a random image from the images folder
file_names = natsorted([x for x in os.listdir(IMAGE_DIR) if x[-5] == '0'])
images = [skimage.io.imread(IMAGE_DIR + name) for name in file_names]

features = {}

# Run detection
for name, image in zip(file_names, images):
	results = model.detect([image], verbose=1)
	r = results[0]
	regions = r['shared'][0]
	best_region = regions[np.argmax(np.asarray([np.linalg.norm(x) for x in regions]))]
	features[name] = best_region

pickle.dump(features, open(IMAGE_DIR + 'features.pkl'), protocol=2)
