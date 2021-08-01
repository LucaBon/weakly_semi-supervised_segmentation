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

The network returns two outputs: 
* one for the pixel-wise classification and
* one for the multiclass image classification




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
on test set are saved in `EncDecUnpool_pixel_labels_epoch15_loss_0.510010"`. The checkpoint can be 
downloaded from: [download checkpoint]("https://anonfiles.com/Lfa6L39fu5/EncDecUnpool_pixel_labels_epoch15_loss_0_510010")
The over-fitting has been reduced but still present. The performances on test set 
increased:

On test set (with dropout)

Mean IoU: 0.596

class IoUs: 
* "Impervious surfaces": 0.715
* "Building": 0.760
* "Low vegetation": 0.487
* "Tree": 0.656
* "Car": 0.363

---------

**Task (ii)**
I haven't implemented the solution.

The idea is to train first with pixel-level labels as in Task (i). 
Evaluate the output of the log_softmax layer giving as input the N2 dataset.
The output has size [batch_size, n_classes, 200, 200] with values ranging in
[-inf, 0]. Set a threshold for each class. Using the image labels associated,
select just the activation maps associated with the classes present in the
image and use the threshold to build a mask for that class. A criteria should
be defined to compose together the masks to obtain a pseudo-pixel-level label
for each sample. Save the pseudo-pixel-level labels and use them to train the
network. In the end, refine the result training again on the N1 pixel-level
labels for some epoch.

Eventually, it is possible to freeze the weights for all layers except the 
dense ones and train the multi-class classifier.




