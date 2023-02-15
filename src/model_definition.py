from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merging import add
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
import numpy as np
from text_data_preparation import load_doc, photo_to_description_mapping
from tokenization import create_tokenizer, max_lengthTEMP
from loading_data import load_clean_descriptions, load_photo_features, load_photo_identifiers

filename = './kaggle/input/flickr8k./captions.txt'

# Loading descriptions
doc = load_doc(filename)

# Parsing descriptions
descriptions = photo_to_description_mapping(doc)
print('Loaded: %d ' % len(descriptions))

train = load_photo_identifiers(filename)
print('Dataset: ',len(train))

train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=', len(train_descriptions))

train_features = load_photo_features('features.pkl', train)
print('Photos: train=', len(train_features))

tokenizer = create_tokenizer(train_descriptions)
max_length = max_lengthTEMP(descriptions)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: ', vocab_size)

def data_generator(descriptions, photos, tokenizer, max_length):
    while True:
        for key, description_list in descriptions.items():
            if key in photos.keys():
                photo = photos[key][0]
                input_img, input_seq, output_word = create_sequence(tokenizer,
                max_length, description_list, photo)
                yield[[input_img, input_seq], output_word]

def create_sequence(tokenizer, max_length, description_list, photo):
    X1, X2, y = list(), list(), list()

    for description in description_list:
        seq = tokenizer.texts_to_sequences(description)[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_length):
    
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    print(inputs1)
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # sequence model
    inputs2 = Input(shape=(max_length,))
    print(inputs2)
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # summarize model
    print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True)
    
    return model