import os
from PIL import Image
import glob
import cv2
import numpy as np

########## How to skip 3rd process? turn on the line 33, 184, turn off the line 163~180, 250

##### Define variables

# Lists for image
ref_img_list = [] #image with reference line to rotate
rot_img_list = [] #rotated image with reference line
bin_img_list = [] #binary image
fin_img_list = [] #final_cropped image

# File routes specification
original_path = "../data/team7/test-MV540-H/"
# (1) Croped data
crop_path = "../data/crop_data/"
if not os.path.isdir(crop_path):
    os.mkdir(crop_path)
# (2) Rotated data
rot_path = "../data/rotate_data/" 
if not os.path.isdir(rot_path):
    os.mkdir(rot_path)
# (3) Binarized data
bin_path = "../data/binary_data/" 
if not os.path.isdir(bin_path):
    os.mkdir(bin_path)
# (4) Final (Re-cropped) data
final_path = "../data/final_data/" 
# final_path = "../data/final_binary_data/" # TURN IN ON TO SKIP 3rd PROCESS
if not os.path.isdir(final_path):
    os.mkdir(final_path)
if not os.path.isdir(final_path+"0/"):
    os.mkdir(final_path+"0/")
if not os.path.isdir(final_path+"1/"):
    os.mkdir(final_path+"1/")


##### Image processing

##### (1) IMAGE CROP #####

files = glob.glob(original_path + '*')

for idx, file in enumerate(files):
    fname, ext = os.path.splitext(file)     

    if ext in ['.bmp']:
        img = Image.open(file)
        width, height = img.size
        crop_image = img.crop((400,300,800,700))

        ##### (1) save to files #####
        if idx+1 <= 9:
            crop_image.save(crop_path + '0' + str(idx+1) + '.bmp')
        else :
            crop_image.save(crop_path + str(idx+1) + '.bmp')
    print(file, "done")

##### (2-1) FIND REFERENCE LINES TO ROTATES #####

files = glob.glob(crop_path + '*')

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

        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2) # Draw lines

        y1 = -y1 # For the change of origin
        y2 = -y2 # For the change of origin

        if x1 == x2: # In case of same x
            x2 += 0.01
        if y1 == y2: # In case of same y
            y2 += 0.01

        # Find theta from the slope
        angle = np.arctan((y2-y1)/(x2-x1))*57.296

        if - 90 < angle < - 30 :
            angle += 180
        if  -30 < angle < 0 :
            angle += 360

        print(file, angle)
        break

    # Save to the list
    ref_img_list.append(img)

    ##### (2-2) IMAGE ROTATION #####

    if idx+1 <= 9:
        img = Image.open(crop_path+ '0' + str(idx+1) + '.bmp')
    else :
        img = Image.open(crop_path + str(idx+1) +'.bmp')
    
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
    
    # Save to the list
    rot_img_list.append(img_new)

    print(file, "done")

##### (3) FROM GRAYSCALE TO BINARY IMAGE #####  # TURN IT OFF TO SKIP 3rd PROCESS

files = glob.glob(rot_path + '*')

for idx, file in enumerate(files):
    img_new = cv2.imread(file)
    img_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    # img_bin = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
    
    # Save to the list
    bin_img_list.append(img_bin)

    ##### (3) save to files #####
    img_bin = Image.fromarray(img_bin) # turn to img
    if idx+1 <= 9:
        img_bin.save(bin_path+'0' + str(idx+1) +'.bmp')
    else :
        img_bin.save(bin_path + str(idx+1) +'.bmp')

##### (4) CROP AGAIN #####

files = glob.glob(bin_path + '*')
# files = glob.glob(rot_path + '*') # TURN IN ON TO SKIP 3rd PROCESS

for idx, file in enumerate(files):
    fname, ext = os.path.splitext(file)     
    img = Image.open(file)
    width, height = img.size
    crop_image = img.crop((65,65,375,375))

    ##### (4) save to files #####
    if idx+1 <= 9:
        # 1-> assignable cause 0-> no assignable cause
        if idx+1 in [18,23,24,26,27,28,29,30,35,36,37,38,39,40,42,44,45,46,47,49]:
            crop_image.save(final_path + '1/0' + str(idx+1) + '.bmp')
        else: 
            crop_image.save(final_path + '0/0' + str(idx+1) + '.bmp')            
    else :
        # 1-> assignable cause 0-> no assignable cause
        if idx+1 in [18,23,24,26,27,28,29,30,35,36,37,38,39,40,42,44,45,46,47,49]:
            crop_image.save(final_path + '1/' + str(idx+1) + '.bmp')
        else: 
            crop_image.save(final_path + '0/' + str(idx+1) + '.bmp')

    # Save to the list
    img = img.convert("L") # Turn to numpy type to put in the list
    img = np.array(img)
    fin_img_list.append(img)
    print(file, "done")

##### Display all images

# Define functions

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

# Start displaying

height = 150 # Squeeze to show all
width = 150 # Squeeze to show all

depth = 3 # With color space
dstimage = create_image_multiple(height, width, depth, 5, 10)
showall(ref_img_list) # Reference line image list
showall(rot_img_list) # Rotated image list

depth = 1 # Without have color space
showall(bin_img_list) # Binarized image list # TURN IT OFF TO SKIP 3rd PROCESS
showall(fin_img_list) # Finally cropped image list