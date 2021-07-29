# weakly_semi-supervised_segmentation

To run the code a **GPU** is necessary.

**docker** and **nvidia-docker** should be installed.

###Environment (see Dockerfile and requirements.txt):
CUDA 9.2
python 3.6
torch 1.5.0
torchvision 0.6.0 

### Build the image
From the root folder

* ```docker build -t weakly_sup_seg .```


### Run the image

* ```docker run --rm -it -v "path_to_project":/app -v "path_to_images":/images - "path_to_labels":/labels weakly_sup_seg bash```

### Training
Inside docker container run: