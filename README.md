# SP-VDSR
SP-VDSR is a Residual Convolutional Neural Network with Depth-to-Space Sub-Pixel Convolution, it's based on [VDSR](https://arxiv.org/abs/1511.04587) and [ESPCN](https://arxiv.org/abs/1609.05158). The network's architecture is shown in the picture below.

<img src="https://raw.githubusercontent.com/Artoriuz/sp-vdsr/master/images/architecture.png" width="500">

For more information, refer to the [paper](https://raw.githubusercontent.com/Artoriuz/sp-vdsr/master/paper/SP-VDSR.pdf).

## Usage Instructions
This repository is still in a very rustic state, and the model itself isn't anything worth using. I highly recommend [RealSR](https://github.com/nihui/realsr-ncnn-vulkan) and [ISR](https://github.com/idealo/image-super-resolution).

If you still want to check SP-VDSR though, you'll have to parametise and point the directories to your images in the code itself. Standard Keras APIs are used so inference is done through a simple model.predict, but currently a 
Matlab .mat file is outputed instead of an image.

The model only supports grayscale images (single channel, luminance only) and it's currently hard-coded to receive 960x540 images and upscale them with a 2x scaling factor (back to 1920x1080). This, however, can be easily changed in the code itself for training or inference.

Imagemagick was used to convert images to grayscale PNGs and to upscale them with the Lanczos kernel, but any other program would work just fine.

## Results
The following image shows preliminary results on a model trained with 32 line-art images.

<img src="https://raw.githubusercontent.com/Artoriuz/sp-vdsr/master/images/results.png">