import os
import string

print(os.listdir('./kaggle/input/flickr8k'))

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def photo_to_description_mapping(descriptions):
    description_mapping = dict()
    for line in descriptions.split('\n'):
        word = line.split()
        #Skipping lines with length less than 2
        if len(line) < 2:
            continue
        #first word image ID and rest is image description
        img_id, img_desc = word[0], word[1:]
        #Removing image extension
        img_id = img_id.split('.')[0]
        img_desc = ' '.join(img_desc)
        # There are multiple descriptions per image, 
        # hence, corresponding to every image identifier in the dictionary, there is a list of description
        # if the list does not exist then we need to create it
        if img_id not in description_mapping:
            description_mapping[img_id] = list()
        
        description_mapping[img_id].append(img_desc)

    return description_mapping

def clean_descriptions(description_mapping):

    table = str.maketrans('', '', string.punctuation)

    for key, descriptions in description_mapping.items():
        for i in range(len(descriptions)):
            description = descriptions[i]
            description = description.split()
            #To lower case
            description = [word.lower() for word in description]
            #Remove punctuations
            description = [word.translate(table) for word in description]
            #Removing word of length =1
            description = [word for word  in description if len(word)>1]
            #Removing numbers
            description = [word for word in description if word.isalpha()]
            #List-->String
            descriptions[i] = ' '.join(description)
        
    return descriptions

def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def save_descriptions(descriptions, filename):
    line = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            line.append(key + ' ' + desc)
    data = '\n'.join(line)
    file = open(filename, 'w')
    file.write(data)
    file.close()


filename = './kaggle/input/flickr8k/captions.txt'

# Loading descriptions
doc = load_doc(filename)

# Parsing descriptions
descriptions = photo_to_description_mapping(doc)
print('Loaded: %d ' % len(descriptions))

#Cleaning descriptions
clean_descriptions(descriptions)

# Summarizing the vocabulary
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

# Saving to the file
save_descriptions(descriptions, 'descriptions.txt')