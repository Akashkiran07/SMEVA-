import cv2
import numpy as np
from PIL import Image
import pytesseract

myconfig = r"--psm 11 --oem 3"

def ocr(image):
    response = pytesseract.image_to_string(image, config= myconfig)
    return response




def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def thersholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] 


def ocr_image():
    img = cv2.imread('IMAGE.jpg')
    #img = grayscale(img)
    #img = thersholding(img)
    #img = remove_noise(img)
    target_color = np.array([255, 0, 0]) 
    #target_color1 = np.array([255,255,255]) # Example target color: pure red
    matches = np.sum(np.all(img == target_color, axis=0))
    #matches1 = np.sum(np.all(img == target_color1, axis=0))
    if  matches==0:
        text = ocr(img)
        text1=text.lower()
        print(text1)
    else:
        text=""    
        text1=text.lower()
     #print(text)
    
    return text1

