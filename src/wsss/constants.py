LABEL_NAMES = ["Impervious surfaces",
               "Building",
               "Low vegetation",
               "Tree",
               "Car",
               "Clutter"]

N_CLASSES = 6
# Clutter is considered noise: it is predicted but excluded from tags and metrics
CLUTTER = 5
N_TAG_CLASSES = 5
EVAL_CLASSES = list(range(N_TAG_CLASSES))
CAR = 4

IGNORE_INDEX = 255

COLOR_MAPPING = {0: (255, 255, 255),  # Impervious surfaces (WHITE)
                 1: (0, 0, 255),  # Building (BLUE)
                 2: (0, 255, 255),  # Low vegetation (TURQUOISE)
                 3: (0, 255, 0),  # Tree (GREEN)
                 4: (255, 255, 0),  # Car (YELLOW)
                 5: (255, 0, 0),  # Clutter/background (RED)
                 }

# Raw ISPRS file names
IMAGE_NAME_FORMAT = "top_mosaic_09cm_area{}.tif"
ERODED_LABEL_NAME_FORMAT = "top_mosaic_09cm_area{}_noBoundary.tif"

# ImageNet statistics, used for every pre-trained encoder (IRRG fed as pseudo-RGB)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
