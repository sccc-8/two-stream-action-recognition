# Two-Stream Reproduce

## Introduction

The video recognition research is mostly inspired by the advances in image recognition methods, and a large amount of video action recognition methods are based on high-dimensional encodings of local spatio-temporal features as presented in Histogram of Oriented Gradients (HOG) and Histogram of Optical Flow (HOF).

This paper proposes a new two-stream architecture to cope with the challenge of action recognition in video data. This architecture shows a competitive performance in video classification tasks, and it also offers a potential research direction in this field that training a temporal stream network on optical flow is a feasible and good method to deal with these tasks.

This blog aims to describe our attempt to reproduce certain aspects of the paper ‘Two-Stream Convolutional Networks for Action Recognition in Videos’ by Karen Simonyan and Andrew Zisserman [1]. We chose to reproduce the results obtained by this paper and extend work by using different data augmentation methods to realize the effect of the processed dataset. Besides, we attempted to perform a hyperparameter to see how the learning rate and batch size influence the performance of the network.

## Two-Stream Architecture

![](D:\reproduce\two-stream-action-recognition\blog\conv.png)

Due to the fact that deep Convolutional Networks (ConvNets) performs well on image classification tasks, Karen Simonyan [1] aimed to use this type of neural network to achieve action recognition in video data. However, if we still choose to use consecutive stacked video frames as input to the network intuitively, it is hard to get a good result. The reason is that in this case, the network needs to learn spatio-temporal motion-dependent features implicitly, which may not be feasible for a certain network. Therefore, a new architecture was proposed to get higher performance in action recognition tasks in videos, which is called two-stream networks. 

![](D:\reproduce\two-stream-action-recognition\blog\optical flow.png)

Generally, this network consists of two separate recognition streams (spatial and temporal), which are then combined by late fusion. This idea is based on the assumption that videos can be decomposed into spatial and temporal parts. The spatial part is in the form of individual frames of the video, which contains the information about scenes and objects. The temporal part is in the form of motion across multiple frames, which shows the movement of the observer and the objects. Accordingly, the spatial stream of the model can recognize actions from still video frames, since some actions are strongly related with certain objects. The temporal stream is able to perform action recognition in the form of dense optical flow. The input of the temporal stream can be seen as a set of displacement vector fields between several consecutive frames. It can display the motion between video frames explicitly, which makes the recognition task easier than previous methods.

## Dataset: UCF101

### Original Dataset

Our architecture for spatial and temporal networks is trained and tested on UCF-101, a challenging dataset of actions. It is an action recognition data set of realistic action videos of people, which has 101 kinds of different action classes, over 13 thousand clips and 27 hours of video data. It is extended from UCF-50. 
UCF-101 includes action classes of 5 types: human-object interaction, body-motion only, human-human interaction, playing musical instruments, and sports. Each action class consists of 25 groups with 4 to 7 motion clips. All video clips have fixed frame rate of 25 FPS and resolution of 320*240. 

### Preprocessed Dataset

We use the preprocessed data of UCF-101 as input data for spatial network and temporal network from [here](https://github.com/feichtenhofer/twostreamfusion ) concerning the intensive computation to process this large volume of dataset ourselves (over 60 GB in total).

For spatial input data, RGB frames are extracted from each video with a sampling rate of 10 and save them as .jpg images. The picture below shows the RGB frames from clip 01 of group 01 of “ApplyEyeMakeup” action class.

![eye](D:\reproduce\two-stream-action-recognition\blog\eye.png)

For motion input data, 2-channel optical flow images are generated through [Flownet2.0](https://github.com/lmb-freiburg/flownet2-docker)  and the x and y channels are saved as .jpg images. To compute optical flow, multiple FlowNets are used in the complete architecture to compute the optical flow of two input images. Brightness Error is the difference between the first image and the second image computed with previously estimated flow.

![fln](D:\reproduce\two-stream-action-recognition\blog\fln.png)

The picture below presents 9 consecutive  preprocessed optical flow image from clip 01 of group 01 of “ApplyEyeMakeup” action class. The bright contour in the figure indicates the moving hands in the video.

![](D:\reproduce\two-stream-action-recognition\blog\1 (1).png)

## Original Code

Since the code of this paper is not available, and this type of deep ConvNet is very difficult for us to build and train on our own without enough guidance and computational resources (the original paper says that a single temporal Convnet takes 1 day for training on a system with 4 NVIDIA Titan cards). Finally, we chose to use the code provided by this [repository](https://github.com/jeffreyyihuang/two-stream-action-recognition), which is a two-stream architecture using spatial and motion stream CNN with Resnet101. However, the readme file of this repo does not provide any information about the environment it operates, we hardly ran the default code when we tried to transfer it to Google Colab. We also observed that most of those *.py* files in this repo are written in python 2, when we put it in a python 3 environment, many syntax errors occurred at once. Besides, we met some issues related to ‘dataloader’, the reason might be that the contributor of this repo preprocessed input data himself instead of using the provided data. Thus, we spent a lot of time debugging the provided code and building required environment. To directly save the modified code, we uploaded them into Google Drive and wrote a script to get them working in a Jupyter Notebook environment. Another problem we met is how to get access to those preprocessed datasets. It is time-consuming to download them to Google Colab due to the volume of the data. Also, how to unzip these data is also a big problem because the mechanism inside Google Drive is that it can not unzip as fast as our local laptop and sometimes it will lose connection automatically. To solve these issues, extra space of Google Drive was purchased to have enough space to save the zip file of dataset. However, we still need to copy the zip file to the root menu of Google Colab (‘/content’) and unzip them every time we implement our experiments.

*Spatial_cnn.py* defines the training process for the spatial CNN. ResNet of torch.utils.model_zoo is used for training both networks. Here ResNet-101, which is 101-layer deep, is pre-trained with ImageNet and then used for our RGB dataset. Firstly, the spatial data loader is prepared, which is defined in spatial_dataloader.py. Then the resnet-101 model is trained, with the output computed and loss evaluated in a mini-batch. The techniques in Temporal Segment network are used, and 3 frames are selected randomly from each video. Then the consensus of the frames is derived as prediction for calculating loss of that video level. In the architecture, Stochastic Gradient Descent (SGD) of torch.optim is used as the optimizer. Cross-entropy loss is used as the criterion to measure the performance of classification. ReduceLROnPlateau of torch.optim.lr_scheduler is used for dynamic learning rate reduction when a metric has stopped improving in the training.

Similarly, the training of motion CNN has a similar structure as the spatial CNN. In every mini-batch, 64 videos (batch size) are selected from 9537 training videos. one stacked optical flow in each video is also randomly selected.

The data augmentation and normalization for training is described in the dataloader of each network. Transformations such as random cropping, random flipping horizontally of images are used as data augmentation techniques. The normalization uses the mean and standard deviation of pre-trained ImageNet. 

## Experiments and Results

| Method                                                       | Prec@1 |
| ------------------------------------------------------------ | ------ |
| Spatial Stream ConvNet [1]                                   | 73.0%  |
| Spatial Stream (pretrained model)                            | 81.2%  |
| pretrained + further training (batch_size=25, lr=5e-4)       | 80.4%  |
| pretrained + further training (batch_size=32, lr=0.01)       | 79.1%  |
| pretrained + further training (batch_size=32, lr=1e-4)       | 78.5%  |
| pretrained + CenterCrop + VerticalFlip                       | 80.0%  |
| pretrained + CenterCrop + VerticalFlip + ColorJitter(br=0.5, contract=0.5) | 81.9%  |
| pretrained + CenterCrop + HorizontalFlip + ColorJitter(br=0.5, contract=0.5) | 81.9%  |



## Discussion