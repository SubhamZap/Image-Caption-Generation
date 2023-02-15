# Image-Caption-Generation

![Image with Captions](https://user-images.githubusercontent.com/96906297/218967741-2323b455-d1fe-44d4-961f-3547498d459e.png)


## 1. Dataset
There are around 8000 images and captions corresponding to the images. Each image have multiple captions.

## 2. How to go about this problem?
We have images and captions corresponding to the images but we don't know how to make use of it to generate captions for new incoming images.

We need a model which might take a image and with some computation magic it provide us the caption.

If you know a bit about deep learning models, it take images in the form of pixels and return output depending on the use-cases. Here, we need both images and texts as input for the model which will provide output with its ML magic.

## 3. How the model/machine will understand features of the image and the text corresponding to the features?
For the above problem we need a pre-trained model which can provide us the feature vector of all images. In our problem statement, we used VGG16 model to extract feature embedding of a image but there are other models which can do the task (like DenseNet201 or ResNet50 etc). It might take few minutes depending on the computational power of your device. We have saved our feature vector so that we don't spend time again-and-again training it.
The caption corresponding to its images are slso saved.

## 4. Define our model
The model takes two inputs - feature vector and text embeddings. The feature vector is of size 4096. The text embedding is designed such that the input of first word will predict the output of next word. The text embedding is passed through LSTM. The output of both input vectors are merged and final output is generated with a softmax activation function.

## 5. Feature engineering of input data
The image ID from the feature vector is mapped with the captions data. Each image ID can have more than one caption. Each caption is added with a prefix and suffix and then it was converted to sequences. Three lists are declared - one for feature vector, second for first word and the third for corresponding word. The first two list are fed to the model as input and last list as output.
