import cv2
import numpy as np

def find_min(vector):
    min_element = 255
    for i in range(len(vector)):
        if min_element>vector[i]:
            min_element = vector[i]
            number = i
    return min_element,number

def find_median_mxn(vector):
    for times in range( len(vector)//2+1 ):  
        min_element,number = find_min(vector[times:])      
        tmp = vector[times]
        vector[times] = min_element
        vector[number+times]= tmp
#    print(tmp)
    return min_element
    
def medianBlur(img, kernel=[3,3], padding_way='ZERO'):
    height,width = img.shape
    height_expand = int(kernel[0]/2)
    width_expand = int(kernel[1]/2)
    img_padding = np.zeros((height+kernel[0],width+kernel[1]),np.uint8)
    vector=[]
    if padding_way=='ZERO':
        img_padding[height_expand:height+height_expand,width_expand:width+width_expand]=img[:,:]
#    elif padding_way=='REPLICA':
#        for rows in range(height_expand):
#            img_padding[rows][cols]=img[][]
    img_blur = np.zeros((height,width),np.uint8)
    for height_loop in range(height):
        for width_loop in range(width):
            sort_vector = img_padding[height_loop:height_loop+kernel[0],width_loop:width_loop+kernel[1]]
            for i in range(kernel[0]):
                for j in range(kernel[1]):
                    vector.append(sort_vector[i][j])
            median_element = find_median_mxn(vector)
            vector.clear()
            img_blur[height_loop,width_loop] = median_element
    return img_blur
import cv2
import numpy as np
img_bgr = cv2.imread('./picture/test/jiaoyan.png',0)
cv2.imshow('img_original',img_bgr)
img_blur = medianBlur(img_bgr,[5,5])
cv2.imshow('img_blur',img_blur)
key=cv2.waitKey()
if key == 32:
    cv2.destroyAllWindows()
