# pytorch-two-stream-cnn-ucf101
We use a spatial and motion stream cnn with ResNet101 as baseline for modeling video information in UCF101 dataset.

## 1. Data preprocessing
  ### 1.1 Spatial data -> rgb frames
  We extract RGB frames from each video in UCF101 dataset with sampling rate: 10, save as .jpg file in disk which cost about 5.9G.
  ### 1.2 Motion data -> stacked optical flow images
  In motion stream, we use two methods to get the desire optical flow data. First method is download the preprocessed tvl1 optical flow dataset directly from https://github.com/feichtenhofer/twostreamfusion>.
    