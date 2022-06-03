from PIL import Image
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Find route
data_dir = "../data/final_data/"
categories = ["0","1"]
nb_classes = len(categories)

# Confirm image size
image_w = 310
image_h = 310
pixels = image_w * image_h

# List for image(X) and label(Y)
X = []
Y = []

# Load all image for making dataframe
for index, cat in enumerate(categories):
    image_dir = data_dir + cat + "/"
    print(image_dir)
    files = glob.glob(image_dir+"*")
    print('files ->', files)
    for idx, file in enumerate(files):
        img = Image.open(file)
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(index)
        if idx % 50 == 0:
            print(idx, "\n", data)
X = np.array(X) # image - first 30 is correct, last 20 is wrong
Y = np.array(Y) # label - first 30 is 0, last 20 is 1
# print(X)
# print(Y)

# Reshape images in a one row
X = X.reshape(50,96100)
print(X.shape)

# Into dataframe
data = pd.DataFrame(X)

# Input normalization - makes no differences AT ALL
# data /= 255

# Find pc ratio
pca = PCA()
pca.fit(data)
pcscore = pca.transform(data)
PC_ratio = pca.explained_variance_ratio_
pc_ratio_df = pd.DataFrame(PC_ratio)
print(pc_ratio_df)

# Show cumulative pc ratio
sum = 0
for i in range(len(pc_ratio_df)):
    sum += pc_ratio_df[0][i]
    print(i+1,"th: ",sum)
# pc_ratio_df.plot(kind='bar')
# plt.show()

# For 85% how many pc to obtain? 30
NUM_OF_COMP = 30

# Do pca using 30 pc
pca = PCA(n_components=NUM_OF_COMP)
pcscore = pca.fit_transform(data) #fit and transform

# Make column name and print scores
columns = []
for i in range(NUM_OF_COMP):
    columns.append("pc"+str(i+1))
# print(columns)
pcscore = pd.DataFrame(data=pcscore, columns=columns)
print(pcscore)

# For evaluation with pca split data into 80% train set 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
pca = PCA()
pca.fit(X_train)
train_score = pca.transform(X_train)
test_score = pca.transform(X_test)

# Make regression model with train set
clf2 = LogisticRegression(max_iter=2000, random_state=0, multi_class='multinomial', solver='sag')
clf2.fit(train_score[:,:40], y_train)
pred2 = clf2.predict(test_score[:,:40])

# Evaluate and get accuracy for test set
for i in range(len(y_test)):
    if y_test[i] == pred2[i]:
        print(y_test[i], "vs", pred2[i], "-> O")
    else:
        print(y_test[i], "vs", pred2[i], "-> X")
cf2 = confusion_matrix(y_test, pred2)
print(cf2)
print("ACCURACY:", accuracy_score(y_test, pred2))