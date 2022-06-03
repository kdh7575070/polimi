from PIL import Image
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA

##### Before pca

# Find route
data_dir = "../data/final_data/"
categories = ["0","1"]
nb_classes = len(categories)

# Confirm image size - reduction to apply it to minitab
# Also, 310*310 Image is too big for finding eigenvector matrix using python

image_w = 50
image_h = 50
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
X = np.array(X) # image - first 30 is no assignable defect, last 20 is assignable defect
Y = np.array(Y) # label - first 30 is 0, last 20 is 1
# print(X)
# print(Y)

# Reshape images in a one row
X = X.reshape(50,2500)
print(X.shape)

# Into Dataframe
data = pd.DataFrame(X)

# Input normalization - makes no differences AT ALL
# data /= 255

# Use only no assignable defect data for pca
data_30 = data[:30].copy()
print(data_30)

# Standardize this data for correlation (standardize + covariance => correlation)
mean_df = data_30.mean() # mean of correct data
std_df = data_30.std() # stdD of correct data
for i in data_30: 
    if mean_df[i] == 0 and std_df[i] == 0: # we cannot devide 0 by 0
        pass
    else:
        data_30[i] = ( data_30[i] - mean_df[i] ) / std_df[i]

# Find covariance matrix, eigenvalues and eigenvectors
cov_mat = np.cov(data_30.T)
print(cov_mat)
eig_vals, eig_vecs = eig(cov_mat) # problem -> it is not decreasing order

# Make it decreasing order
idx = eig_vals.argsort()[::-1]   
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:,idx]
print(eig_vals) 
print(eig_vecs)


##### Pca correlation - python use correlation

# Find pc ratio
pca = PCA()
pcscore = pca.fit_transform(data_30) # combination of [pca.fit(data2)] & [pcscore = pca.transform(data2)]
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

# For 80% how many pc to obtain? 12
# For 90% how many pc to obtain? 16
NUM_OF_COMP = 12
# NUM_OF_COMP = 16

# Create new standardized data of original data for finding new scores for all data later 
newsc = data.copy()
for i in newsc: 
    if mean_df[i] == 0 and std_df[i] == 0:
        pass
    else:
        newsc[i] = ( newsc[i] - mean_df[i] ) / std_df[i]
print(newsc)


##### Pca using num of pc found above

# Find the new scores of all the data using 12pc
# Instead of pca, we manually did it because python doesn't support for that function
# (the original data * the eigenvectors computed only on the first 30 data)
print(newsc.shape)
print(eig_vecs[:,:NUM_OF_COMP].shape)
newscore = newsc.to_numpy().dot(eig_vecs[:,:NUM_OF_COMP])
print(newscore)

# Make column name and print scores
columns = []
for i in range(NUM_OF_COMP):
    columns.append("pc"+str(i+1))
# print(columns)
showpc = pd.DataFrame(data=newscore, columns=columns)
print(showpc)

# Find hat - with new scores and eigenvectors 
# Instead of pca, we manually did it because python doesn't support for that function
# ( newscore * transpose of the eigenvectors computed only on the first 30 data)
hat = newscore.dot(eig_vecs[:,:NUM_OF_COMP].T)
print(hat)

# Reconstruct data from back-standardization because we used correlation for the first pca
back = pd.DataFrame(hat)
for i in back: 
    back[i] = ( back[i] * std_df[i] ) + mean_df[i]
# print(back)


##### Post processing, saving files and visualization

# Result type is complex number -> change to real number
back = back.to_numpy()
back = [c.real for c in back]
back = pd.DataFrame(back)

# Put reconstructed data in csv file and print it
back.to_csv('finalfinal.csv')
# back.to_csv('finalfinal_90.csv')
print(back)

# Make difference matrix between original data and reconstructed data
diff = data - back

# Put difference matrix in csv file and print it
back.to_csv('difference.csv')
# back.to_csv('difference_90.csv')
print(diff)

# Show it visually
rows = 5
columns = 10
back = back.to_numpy()
back = back.reshape(50,50,50)
for idx, i in enumerate(back) : 
    image_index = idx + 1     # image index 
    ttitle = "Image{}".format(image_index) # image title
    plt.subplot(rows, columns, image_index) # subplot 
    plt.title(ttitle)   # title 
    # // plt.axis('off')
    plt.xticks([])  # x = None 
    plt.yticks([])  # y = None
    plt.imshow(i)  
plt.show()