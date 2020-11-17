import cv2
import imutils
import statistics
import numpy as np
import math
image = cv2.imread("z9n.jpg")
kernel = np.ones((2,2), np.uint8)
image = cv2.resize(image, (400, 400))
image = cv2.dilate(image, kernel, iterations=1)
iq = image.copy()
def get_x(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)
    for (i, c) in enumerate(cnts[2:]):
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        approx_area = cv2.contourArea(approx)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and approx_area/hull_area > 0.3 and approx_area/hull_area  < 0.9:
            cv2.drawContours(mask, [c], 0, (255), 1)
    return mask




def extract(img):

    mask = np.zeros(img.shape[:2], np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)
    for (i, c) in enumerate(cnts[2:]):
        cv2.drawContours(mask, [c], 0, (255), 1)
    return mask

def extract2(img, ang):
    pass



def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


boards = []
# load the tic-tac-toe image and convert it to grayscale

image1 = image
#image = cv2.imread("10.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray)
gray = 255 -gray
gray= cv2.Canny(gray, 100, 200)
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#print(cv2.boundingRect(cnts[0])[3])
cnts = sorted(cnts, key = lambda c : (cv2.boundingRect(c) [2] * cv2.boundingRect(c) [3] ), reverse= True)
"""for (i, c) in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    if i ==-1:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)"""
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
#x, y, w, h = cv2.boundingRect(cnts[0])
#cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("Output3", image1)
while cnts:
    box = cv2.boxPoints(cv2.minAreaRect(cnts[0]))
    #print(box[1])
    box = np.int0(box)
    cv2.drawContours(image1, [box], 0, (0, 255, 255), 2)
    x, y, w, h = cv2.boundingRect(cnts[0])
    if (w/h>4 and len(cnts) > 1)  or ((box[1][0]/box[1][1])>4 and len(cnts)>1) :#or cv2.minAreaRect(cnts[0])[2] < 10 and len(cnts) > 1:
        x, y, w, h = cv2.boundingRect(cnts[1])
    elif w/h > 4:
        break



    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:,:] = 255
    mask[y:y + h, x:x + w] = 0
    t = 1


    if w * h > 5000:

        boards.append(gray[int(0.9*y):y + int(h*1.5), int(0.9*x):int(x + 1.4*w)])
        print(w, h)

        mask[y:y + h, x:x + w] = 0
    gray = cv2.bitwise_and(gray, mask)
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key = lambda c : (cv2.boundingRect(c) [2] * cv2.boundingRect(c) [3] ), reverse= True)


#cv2.imshow("Output2", gray)
t = 0
cv2.imshow("out", image)
for i in boards:
    extract(i)
    cv2.imshow(str(t), get_x(i))
    t +=1

gray = cv2.cvtColor(iq, cv2.COLOR_BGR2GRAY)
gray = 255 -gray
gray= cv2.Canny(gray, 100, 200)
cnts = cv2.findContours(gray.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(iq, cnts, -1, (255), 3)
cv2.imshow("ou", iq)
cv2.waitKey(0)