[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/7lKBcjfN)
# Image Denoising Using U-Net Architecture

## Project Purpose

Image denoising is an extremely popular image processing technique, especially in the medical community. Obtaining high-resolution MRI scans can be time consuming and require the patient to remain completely still in order to get detailed scans of the brain; however, this isn't always the case. Patients can be restless, among other things, while in the scanner and the only way to finish faster is by cutting the time to scan in half. Doing so would present low-resolution images with artifacts or noise, a graininess effecting the resolution details of an image. The purpose of this project is to denoise noisy images using the [U-Net](https://arxiv.org/pdf/1505.04597.pdf) model with a goal to see how structurally similar the denoised images predicted by the model are compared to their originals.

## Data Collection and Manipulations

For this project, I gathered over 1,000 cat images using the [Pexels API](https://www.pexels.com). You can see the mining Pexels API code under the [Pexel Image Mining notebook](/Pexel_Image_Mining.ipynb). Each image collected from the API was of various size in resolution, color images, and copyright & royalty free.

Since the images were of various sizes, I resized each image down to 256x256 shape for consistentcy in the model. You can see the resize code under the [Resize Images notebook](./Resize_Images.ipynb). Resizing the images down to 256x256 will allow for easier use on the U-net architecture as I will be using the default architecture from the [original paper](https://arxiv.org/pdf/1505.04597.pdf). To create noisy images, I will be adding Gaussian-distributed noise to each photo. You can see an example visual of how this step is done under the [Cat Noise Test notebook](./Cat_Noise_Test.ipynb).

## U-Net Background and Utilization

A U-Net is a type of convolutional neural network (CNN) architecture published in 2015 by Olaf Ronneberger et al. that is commonly used for image segmentation tasks. However, it has also been adapted for image denoising tasks in other [research](https://stanford.edu/class/ee367/Winter2019/dua_report.pdf). The U-Net architecture is particularly effective for tasks where the input and output have the same spatial dimensions, such as image denoising.

![U-net Architecture](/Users/tevaughnshaw/ml-s24-project-tevaughnshaw/Unet_Architecture.png)
*U-Net Architecture* [image source](https://arxiv.org/pdf/1505.04597.pdf)

The encoder path of the U-Net downsamples the input image through a series of convolutional layers, reducing its spatial dimensions while increasing the number of feature channels. Each downsampling step typically involves convolutional layers followed by pooling layers (e.g., max pooling) to capture hierarchical features of the input image.
The decoder path of the U-Net upsamples the feature maps back to the original input resolution while reducing the number of feature channels. Each upsampling step typically involves transposed convolutional layers (also known as deconvolution or upsampling layers) to gradually reconstruct the spatial information lost during the encoding process. Skip connections are used to concatenate feature maps from the encoder path with corresponding feature maps in the decoder path, helping to preserve fine-grained details during upsampling.
The output layer of the U-Net produces the denoised image. It typically consists of one or more convolutional layers followed by an activation function (e.g., sigmoid or tanh) to constrain the pixel values to a certain range (e.g., 0, 1 for images). The output of the U-Net is a denoised version of the input image with the same spatial dimensions.

## Metrics
### Structural Similarity Index

The Structural Similarity Index (SSIM) is a metric to quantify the perceived similarity between two images. Unlike other metrics like Mean Squared Error (MSE) or Peak-Signal-to-Noise-Ratio (PSNR), which focus on the individual pixel differences independently, SSIM considers how humans perceive visual information in digital imaging. After applying a denoising technique, SSIM is calculated between the original noisy image and the denoised image. There are three factors  considered by SSIM:

- Luminance: measures the brightness similarity between the images. Is measured by averaging all of the pixel values.
- Contrast: measures the similarity in local variations of pixel intensities. Is measured by taking the standard deviation (square root of variance) of all pixel values.
- Structure: evaluates how well the spatial arrangement of pixels is preserved. Is measured by dividing the input signal with its standard deviation.
[source](https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e)

The range of SSIM is from -1 to 1. Being closer to 1 indicates that the denoised image is perceptually similar to the original, meaning it has retained detail while removing noise.

### Benefits of Using SSIM

- Focuses on human perception: SSIM better reflects how we see images compared to metrics like PSNR and MSE.
- Guides denoising process: by monitoring SSIM, it is easy to fine-tune the denoising algorithm and find balance between noise removal and detail preservation.

For a visual look at a SSIM example, see code under the [Structural Similarity Index notebook](./Structural_Similarity_Index.ipynb).

### Peak Signal-to-Noise Ratio

Peak Signal-to-Noise Ratio (PSNR) computes the peak signal-to-noise ratio, in decibels (dB), between two images. This ratio is used as a quality measurement between the original and a compressed image. The higher the PSNR, the better the quality of the compressed, or reconstructed image. The mean-square error (MSE) and the peak signal-to-noise ratio (PSNR) are used to compare image compression quality. The MSE represents the cumulative squared error between the compressed and the original image, whereas PSNR represents a measure of the peak error. The lower the value of MSE, the lower the error.[source](https://www.mathworks.com/help/vision/ref/psnr.html)

A higher PSNR dB value generally indicates better quality, as it implies lower distortion or noise in the image.

### Perceptual Loss

Perceptual loss is a type of loss function used in image generation tasks such as image denoising. Unlike traditional loss functions like Mean Squared Error (MSE), which measure pixel-wise differences between the predicted and ground truth images, perceptual loss focuses on capturing high-level visual features and semantics.

Perceptual loss is calculated by comparing the feature representations of the predicted and ground truth images extracted from a pre-trained deep neural network, often a CNN designed for image classification or feature extraction. The idea is to measure the difference in perceptual content between the generated image and the ground truth image rather than the difference in pixel values directly.

### Further Improvements

Further research on this project could include: 
- Increasing the noise rate
- Experimenting with a larger dataset
- Apply a different type of noise
- Use another convolutional model

## Conclusions

![UnetResults]('unet-denoised-results.png')
*Sample Results from U-Net model*

I learned that the denoised images predicted by the U-Net model were able to retain most details from their ground truths with structural relevance. This is shown as the SSIM score went from 1 to the .80 range, which overall means that there is still some noise artifacts that can be perceived in the denoised images.

Standard MSE and PSNR metrics have been used in past denoising research, but aren't the best to use compared to SSIM and other metrics.

U-Nets have been widely used in the medical field in tasks such as denoising MRI scans to provide clearer images for neurologists to perform further analysis.

