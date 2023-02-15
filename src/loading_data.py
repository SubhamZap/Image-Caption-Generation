import pickle

def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_photo_identifiers(filename):
    file = load_file(filename=filename)
    photos = list()
    for line in file.split('\n'):
        if len(line)<1:
            continue
        identifier = line.split('.')[0]
        photos.append(identifier)
    return set(photos)

def load_clean_descriptions(filename, photos):
    file = load_file(filename=filename)
    descriptions = dict()
    for line in file.split('\n'):
        words = line.split()
        img_id, img_desc = words[0], words[1:]

        if img_id in photos:
            if img_id not in descriptions:
                descriptions[img_id] = list()

            desc = 'startseq' + ' '.join(img_desc) + 'endseq'
            descriptions[img_id].append(desc)
    return descriptions

def load_photo_features(filename, photos):
    all_features = pickle.load(open(filename, 'rb'))
    # print(all_features)
    # print(photos)
    features = {k: all_features[k] for k in photos if k in all_features.keys()}
    # print(features)
    return features