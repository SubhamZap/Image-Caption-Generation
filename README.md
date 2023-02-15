# Image-Caption-Generation

![Image with Captions](https://user-images.githubusercontent.com/96906297/218967281-ba459592-58a2-4568-a763-dfc01121f0cc.png)


## 1. Dataset
There are around 8000 images and captions corresponding to the images. Each image have multiple captions.

## 2. How to go about this problem?
We have images and captions corresponding to the images but we don't know how to make use of it to generate captions for new incoming images.

We need a model which might take a image and with some computation magic it provide us the caption.

If you know a bit about deep learning models, it take images in the form of pixels and return output depending on the use-cases. Here, we need both images and texts as input for the model which will provide output with its ML magic.

## 3. How the model/machine will understand features of the image and the text corresponding to the features?
For the above problem we need a pre-trained model which can provide us the feature vector of all images. In our problem statement, we used VGG16 model to extract feature embedding of a image but there are other models which can do the task (like DenseNet201 or ResNet50 etc). It might take few minutes depending on the computational power of your device. We have saved our feature vector so that we don't spend time again-and-again training it.
The caption corresponding to its images are slso saved.

