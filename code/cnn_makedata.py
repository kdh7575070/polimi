from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

# find routh
data_dir = "../data/final_data/"
categories = ["0","1"]
nb_classes = len(categories)

# confirm image size
image_w = 310
image_h = 310
pixels = image_w * image_h # because it's binary

# list for image(X) and label(Y)
X = []
Y = []

for idx, cat in enumerate(categories):
    # make label
    label = [0 for i in range(nb_classes)]
    label[idx] = 1 # [indexing 1 0]

    # get images and put label on it
    image_dir = data_dir + cat + "/"
    print(image_dir)
    files = glob.glob(image_dir+"*")
    print('files ->', files)
    for idx, file in enumerate(files):
        img = Image.open(file)
        img = img.resize((image_w, image_h))
        # print('img->', img)
        data = np.asarray(img)
        # print('data->', data)
        X.append(data)
        Y.append(label)
        if idx % 50 == 0:
            print(idx, "\n", data)
X = np.array(X)
Y = np.array(Y)

# split data into test and training for validation 
X_train, X_test, y_train, y_test = train_test_split(X, Y) #make validation set: default 20%
xy = (X_train, X_test, y_train, y_test)
np.save("../data/5obj.npy", xy)
print("ok,", len(Y))