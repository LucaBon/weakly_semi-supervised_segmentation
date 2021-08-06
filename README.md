# Weakly semi-supervised segmentation

### Problem description

Pixel level labels are expensive, but bounding boxes or image level class labels are a lot
cheaper and more plentiful. Investigate weak supervision methods for making use of these
additional cheap labels, and attempt to improve accuracy on the pixel level task.


N1 = 3 high-resolution images (10%)

N2 = 23 high-resolution images (70%)

N_validation = 7 high-resolution images (20%)

Train a semantic segmentation model on a satellite imagery dataset in a weakly
supervised setting


##### Dataset

There are 33 high-resolution (approx 1500x2000px) aerial images of the town of Vaihingen, Germany.
Labels are supplied with 1 class per pixel, as follows:
1. Impervious surfaces - WHITE
2. Building - BLUE
3. Low vegetation - TURQUOISE
4. Tree - GREEN
5. Car - YELLOW
6. Clutter/background - RED

Clutter is considered noise and will be neglected


N1 = 3 high-resolution images (10%)

N2 = 23 high-resolution images (70%)

N_validation = 7 high-resolution images (20%)


### Environment (see Dockerfile and requirements.txt):
CUDA 9.2

python 3.6

torch 1.5.0

torchvision 0.6.0 


To run the code a **GPU** is necessary.

**docker** and **nvidia-docker** should be installed.


#### Pre-trained weights
Download ResNet38 weights and put them in folder resnet_weights
[resnet38](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth)


### Build the image
From the root folder

* ```docker build -t weakly_sup_seg .```


### Run the image

* ```docker run --rm -it -v "path_to_project":/app -v "path_to_images":/images - "path_to_labels":/labels weakly_sup_seg bash```

### Training
Inside docker container run:

* ```python3 run_training.py``` (it will execute `training_task_1` method )


### Task
Train and compare semantic segmentation networks, using the following data:

Task (i): N1 pixel level labels;
Task (ii): N1 pixel level labels + N2 crop-level class labels.

### Solution description

EncDecUnpool network based on VGG16 has been designed. It is inspired by Deconvnet
"Learning Deconvolution Network for Semantic Segmentation", H. Noh et al.




### Results

**Task (i)**

To reproduce the training run `training_task_1` method in run_training.py script.
 
The weights that gave the best segmentation results on test set are saved in 
`EncDecUnpool_pixel_labels_epoch13_loss0.692428.pth`. The checkpoint can be 
downloaded from: [download checkpoint](https://anonfiles.com/J3rf7e91ud/EncDecUnpool_pixel_labels_epoch13_loss0.692428_pth)
The performances have been evaluated in terms of Intersection over Union. 


On test set (without dropout):

mean IoU: 0.531

class IoUs: 
* "Impervious surfaces": 0.672
* "Building": 0.709
* "Low vegetation": 0.424
* "Tree": 0.583
* "Car": 0.265

There is over-fitting during training, but I didn't have time to do other experiments. 
Dropout can be added to prevent it. The performances on class "Car" are lower
since it is less represented. To address this problem I weighted more the error
on class car in the loss function but was not enough. A sampling strategy that
favours tiles containing cars can be implemented.

-----------

A Dropout layer has been added at every block in the decoder layers.
A new training has been performed. The weights that gave the best segmentation results
on test set are saved in `EncDecUnpool_pixel_labels_epoch19_loss_0.506415`. The checkpoint can be 
downloaded from: [download checkpoint](https://anonfiles.com/ffecW294ud/EncDecUnpool_pixel_labels_epoch19_loss_0_506415)
The over-fitting has been reduced but still present. The performances on test set 
increased:

On test set (with dropout)

Mean IoU: 0.603

class IoUs: 
* "Impervious surfaces": 0.718
* "Building": 0.772
* "Low vegetation": 0.491
* "Tree": 0.654
* "Car": 0.378

---------

**Task (ii)**
For this task I trained a ResNet38 on the multiclass classification problem and
using the method described in "Single-Stage Semantic Segmentation from Image
Labels" by N. Araslanov et al. I implemented an approach
that produces semantic masks from image-level annotations and outperforms CAMs
in terms of segmentation accuracy.
The semantic masks are then used as pseudo-ground truth for the EncDecUnpool
segmentation network.
The results ...




