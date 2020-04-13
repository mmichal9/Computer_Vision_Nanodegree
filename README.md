# Computer Vision Nanodegree
Computer Vision Nanodegree Notes and Projects [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).


# Nanodegree Overview
For more details about the program please have a look at the Nanodegree [Syllabus](/data/CVND_Syllabus.pdf) or visit [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).


# Projects
### [P1: Facial Keypoint Detection](/projects/facial_keypoint_detection)
Use image processing techniques and deep learning techniques to detect faces in an image and find facial keypoints, such as the position of the eyes, nose, and mouth on a face.


# Notes:
### [Module 1: Classical Computer Vision](/notes/Notes_M1_Classical_Computer_Vision.ipynb)
DOCME

### [PyTorch Templates - CNN MNIST](/notes/Templates_PyTorch_MNIST_CNN.ipynb)
DOCME


# Setup
Setup conda environment for Computer Vision Nanodegree using [official guide](https://github.com/udacity/CVND_Exercises) or instruction below:

```
# 1. Create conda environment and activate it
conda create --name cv-nd python=3.6
activate cv-nd

# 2. Add environment to Jupyter Kernels:
conda install pip
conda install ipykernel
python -m ipykernel install --user --name <cv-nd> --display-name "<cv-nd>"

# 3. Install PyTorch and OpenCV
pip install opencv-python
conda install -c pytorch pytorch
conda install -c pytorch torchvision

# 4. Install other required packages:
pip install -r requirements.txt

```