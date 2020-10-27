import cv2
import matplotlib.pyplot as plt

from utils import *
from darknet import Darknet

cfg_file = "./yolov3.cfg"
weight_file = "./yolov3.weights"
# Location and name of the COCO object classes file
namesfile = "data/coco.names"

m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)

# Set default figure size
plt.rcParams['figure.figsize'] = [24.0, 12.0]

# Load the image
img = cv2.imread('./images/dog.jpg')

# Convert the image to RGB
og_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(og_image, (m.width, m.height))

# Non-Maximal Suppresion (NMS) threshold
nms_thresh = 0.6

# Set the IOU threshold
iou_thresh = 0.4

# Detect objects in the image
boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

print_objects(boxes, class_names)