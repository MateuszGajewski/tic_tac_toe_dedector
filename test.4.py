import cv2
import imutils
import statistics
import numpy as np
import math
image = cv2.imread("10.jpg")
iq = image.copy()
def extract(img):

    mask = np.zeros(img.shape[:2], np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=False)
    cv2.drawContours(mask, cnts, -1, (255), 3)
    return mask

def extract2(img):
    img = 255 -img
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 220, minLineLength=100, maxLineGap=10)
    mask = img.copy()#np.zeros(img.shape[:2], np.uint8)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(mask, pt1, pt2, (0), 3, cv2.LINE_AA)
    return mask


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


boards = []
# load the tic-tac-toe image and convert it to grayscale

image1 = image
cv2.imshow("Oryginalny obraz", image)
#image = cv2.imread("10.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray)
cv2.imshow("To grayscale", gray)
gray = 255 -gray
cv2.imshow("Odwrocenie kolorow", gray)
gray= cv2.Canny(gray, 100, 200)
cv2.imshow("Po zastosowaniu filtra Canny", gray)
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
cv2.imshow("Znalezione kontury", image1)
while cnts:
    box = cv2.boxPoints(cv2.minAreaRect(cnts[0]))

    box = np.int0(box)
    #cv2.drawContours(image1, [box], 0, (0, 255, 255), 2)
    x, y, w, h = cv2.boundingRect(cnts[0])
    if (w/h>4 and len(cnts) > 1) :#or cv2.minAreaRect(cnts[0])[2] < 10 and len(cnts) > 1:
        x, y, w, h = cv2.boundingRect(cnts[1])
    elif w/h > 4:
        break



    cv2.rectangle(iq, (x, y), (x + w, y + h), (0, 0, 255), 2)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:,:] = 255
    mask[y:y + h, x:x + w] = 0
    t = 1


    if w * h > 5000:

        boards.append(gray[int(0.9*y):y + int(h*1.2), int(0.9*x):int(x + 1.2*w)])
        print(w, h)

        mask[y:y + h, x:x + w] = 0
    gray = cv2.bitwise_and(gray, mask)
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key = lambda c : (cv2.boundingRect(c) [2] * cv2.boundingRect(c) [3] ), reverse= True)


#cv2.imshow("Output2", gray)
t = 0
cv2.imshow("Znalezione plansze", iq)
for i in boards:
    extract(i)
    cv2.imshow("Plansz nr:"+ str(t),extract(i))
    t +=1
cv2.imshow("ou", iq)
cv2.waitKey(0)