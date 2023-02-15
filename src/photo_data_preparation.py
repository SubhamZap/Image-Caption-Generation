from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from os import listdir
from keras_preprocessing.image import load_img, img_to_array
import pickle

def extract_features(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs= model.inputs, outputs= model.layers[-1].output)

    print(model.summary())

    features = dict()

    for name in listdir(directory):
        #Image path
        img_path = directory + '/' + name
        #load image
        image = load_img(img_path, target_size=(224, 224))
        #Convert image to array
        img_arr =img_to_array(image)
        #Reshape image for model
        img_reshaped = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
        #Preprocessing image for VGG model
        img_preprocessed = preprocess_input(img_reshaped)
        #Getting features of image
        feature = model.predict(img_preprocessed, verbose = 0)
        #Image ID
        img_id = name.split('.')[0]
        #Feature set
        features[img_id] = feature

    return features

directory = r'./kaggle/input/flickr8k/Images'
features = extract_features(directory=directory)
print("Length of features = %d".format(len(features)))
pickle.dump(features, open('features.pkl', 'wb'))