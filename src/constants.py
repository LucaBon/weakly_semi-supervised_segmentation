import torch

TILE_SIZE = (200, 200)
IMAGE_FOLDER_PATH = "/images"
LABELS_FOLDER_PATH = "/labels"

NAME_FORMAT = "top_mosaic_09cm_area{}.tif"

INPUT_CHANNELS = 3
BATCH_SIZE = 10
N_CLASSES = 6

# car class is weighted more
PIXEL_WEIGHTS = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 10.0, 1.0])

N_CLASSES_IMAGE_LABELS = 5  # clutter is neglected
# car class is weighted more
IMAGE_WEIGHTS = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 10.0])

CACHE = True

# Threshold in percentage ( # pixels_of_that_class / # all_pixels)
# If the percentage of pixel of a specific class is greater than the threshold
# that class is considered present, otherwise it is considered absent
THRESHOLD_IMAGE_LABELS = 0

# if the ground truth y_true is [1,0,0,1,1], while the predicted output
# vector y_hat is [0.7, 0.3, 0.5, 0.6, 0.8], to evaluate performances a
# threshold value is picked for the predicted labels to be accepted as true
MULTI_LABEL_THRESHOLD = 0.5

LABEL_NAMES = ["Impervious surfaces",
               "Building",
               "Low vegetation",
               "Tree",
               "Car",
               "Clutter"]

COLOR_MAPPING = {0: (255, 255, 255),  # Impervious surfaces (WHITE)
                 1: (0, 0, 255),  # Building (BLUE)
                 2: (0, 255, 255),  # Low vegetation (TURQUOISE)
                 3: (0, 255, 0),  # Tree (GREEN)
                 4: (255, 255, 0),  # Car (YELLOW)
                 5: (255, 0, 0),  # Clutter/background (RED)
                 }

INVERSE_COLOR_MAPPING = {v: k for k, v in COLOR_MAPPING.items()}

# Number of epoch performed using just image-level labels
EPOCHS_IMAGE_LABELS = 3
LR_IMAGE_LABELS = 0.001
# Number of epochs performed using pixel level labels
EPOCHS = 20

VGG16_URL = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'

BG_SCORE = 0.1  # background score (only for CAM)
FOCAL_P = 3
FOCAL_LAMBDA = 0.01
PAMR_KERNEL = [1, 2, 4, 8, 12, 24]
PAMR_ITER = 10
SG_PSI = 0.3