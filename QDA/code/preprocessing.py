import os
from PIL import Image
import glob
import cv2
import numpy as np


##### VARIABLES #####

##### IMAGE LIST #####

new_img_list = [] #cropped image with line
rot_img_list = [] #rotated image with line
bin_img_list = [] #binary image
fin_img_list = [] #final_cropped image

##### FILE ROUTE #####

path = "../data/team7/test-MV540-H/"

new_path = "../data/new_data/" 

if not os.path.isdir(new_path):
    os.mkdir(new_path)
    
rot_path = "../data/rotate_data/" 

if not os.path.isdir(rot_path):
    os.mkdir(rot_path)

bin_path = "../data/binary_data/" 

if not os.path.isdir(bin_path):
    os.mkdir(bin_path)

final_path = "../data/final_data/" 

if not os.path.isdir(final_path):
    os.mkdir(final_path)


##### IMAGE PROCESSING #####

##### (1) IMAGE CROP #####

files = glob.glob(path + '*')

for idx, file in enumerate(files):
    fname, ext = os.path.splitext(file)     

    if ext in ['.bmp']:
        img = Image.open(file)
        width, height = img.size
        crop_image = img.crop((400,300,800,700))

        ##### (1) save to files #####
        if idx+1 <= 9:
            crop_image.save(new_path + '0' + str(idx+1) + '.bmp')
        else :
            crop_image.save(new_path + str(idx+1) + '.bmp')

    print(file, "done")

##### (2-1) FIND LINES TO ROTATES #####

files = glob.glob(new_path + '*')

for idx, file in enumerate(files):
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200, L2gradient=True)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    ##### (2-1) CHANGE FROM RHO THETA TO X Y SCALE #####

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

        y1 = -y1 #For the change of origin
        y2 = -y2 #For the change of origin

        if x1 == x2: #In case of same x
            x2 += 0.01
        if y1 == y2: #In case of same y
            y2 += 0.01

        #Find theta from the slope
        angle = np.arctan((y2-y1)/(x2-x1))*57.296

        if - 90 < angle < - 30 :
            angle += 180
        if  -30 < angle < 0 :
            angle += 360
            
        print(file, angle)
        break

    #save to the list for cropped image with line
    new_img_list.append(img)

    ##### (2-2) IMAGE ROTATION #####

    if idx+1 <= 9:
        img = Image.open(new_path+ '0' + str(idx+1) + '.bmp')
    else :
        img = Image.open(new_path + str(idx+1) +'.bmp')
    
    if 360 > angle >= 270:
        rot_img = img.rotate(360-angle)
    elif 270 > angle >= 30:
        rot_img = img.rotate(90-angle)
    elif 30 > angle >= 0:
        rot_img = img.rotate(-angle)

    ##### (2-2) save to files #####
    if idx+1 <= 9:
        rot_img.save(rot_path+'0' + str(idx+1) +'.bmp')
    else :
        rot_img.save(rot_path + str(idx+1) +'.bmp')

##### (2-3) DOUBLECHECK ROTATED IMAGE - REDRAW THE LINE ##### 

files = glob.glob(rot_path + '*')

for idx, file in enumerate(files):

    img_new = cv2.imread(file)
    img_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200, L2gradient=True)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, 95)
    
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img_new, (x1,y1), (x2,y2), (0,0,255), 2)
        break
    
    #save to the list for rotated image with line
    rot_img_list.append(img_new)

    print(file, "done")

##### (3) FROM GRAYSCALE TO BINARY IMAGE #####

files = glob.glob(rot_path + '*')

for idx, file in enumerate(files):
    img_new = cv2.imread(file)
    img_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    # img_bin = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
    
    #save to the list for binary image
    bin_img_list.append(img_bin)

    ##### (3) save to files #####
    img_bin = Image.fromarray(img_bin) # turn to img
    if idx+1 <= 9:
        img_bin.save(bin_path+'0' + str(idx+1) +'.bmp')
    else :
        img_bin.save(bin_path + str(idx+1) +'.bmp')

##### (4) CROP AGAIN #####

files = glob.glob(bin_path + '*')

for idx, file in enumerate(files):
    fname, ext = os.path.splitext(file)     
    img = Image.open(file)
    width, height = img.size
    crop_image = img.crop((65,65,375,375))

    ##### (4) save to files #####
    if idx+1 <= 9:
        crop_image.save(final_path + '0' + str(idx+1) + '.bmp')
    else :
        crop_image.save(final_path + str(idx+1) + '.bmp')

    #save to the list for final cropped image
    img = img.convert("L") #turn to np
    img = np.array(img)
    fin_img_list.append(img)

    print(file, "done")


##### DISPLAY ALL #####

##### DEFINE FUNCTIONS #####

def create_image_multiple(h, w, d, hcout, wcount):
    image = np.zeros((h*hcout, w*wcount,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image

def showMultiImage(dst, src, h, w, d, col, row):
    if  d==3: #3 color
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w] = src[0:h, 0:w]
    elif d==1: # 1 color
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 0] = src[0:h, 0:w]
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 1] = src[0:h, 0:w]
        dst[(col*h):(col*h)+h, (row*w):(row*w)+w, 2] = src[0:h, 0:w]

def showall(img_list):
    for i in range(5):
        for j in range(10):
            img_list[i*10+j] = cv2.resize(img_list[i*10+j], (height,width), interpolation = cv2.INTER_AREA)
            showMultiImage(dstimage, img_list[i*10+j], height, width, depth, i, j)

    cv2.imshow('multiView', dstimage)
    cv2.waitKey(0)
    cv2.destroyWindow('multiView')

##### START DISPLAY #####

height = 150 #squeeze to show all
width = 150 #squeeze to show all
depth = 3 #with color space

dstimage = create_image_multiple(height, width, depth, 5, 10)

showall(new_img_list) #cropped image list

showall(rot_img_list) #rotated image list

depth = 1 #without have color space

showall(bin_img_list) #binaried image list

showall(fin_img_list) #finally cropped image list