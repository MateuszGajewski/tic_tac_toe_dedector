import cv2
import imutils
import statistics
import numpy as np
import math
image = cv2.imread("a8.jpg")
kernel = np.ones((2,2), np.uint8)
#image = cv2.resize(image, (1000, 750))
#image = cv2.dilate(image, kernel, iterations=1)
iq = image.copy()

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    if angle < -45:
        angle = -1*(-90 -angle)

    return img_crop, imutils.rotate_bound(img, -angle)


def get_central_square(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1]), reverse=True)

    curr_area = 0
    curr_index = 0
    b = -1
    if len(cnts) > 1:
        M = cv2.moments(cnts[1])
    else:
        M = cv2.moments(cnts[0])
    cX =0
    cY =0
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    image = cv2.circle(mask, (cX,cY), radius=3, color=(255), thickness=3)
    for (i, c) in enumerate(cnts[2:]):
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        approx_area = cv2.contourArea(approx)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)
        #cv2.drawContours(mask, box, 0, (255), 1)
        if (rect[1][0]*rect[1][1]) != 0 and (approx_area/(rect[1][0]*rect[1][1])) > 0 :
            if  cv2.pointPolygonTest(box, (cX,cY), False)==1 and curr_area< approx_area:
                cv2.drawContours(mask, [c], 0, (255), 1)
                cv2.drawContours(mask, [box], 0, (255), 1)
                curr_area = approx_area
                curr_index = i
                b = [box, rect]
        #cv2.drawContours(mask, cnts[curr_index], 0, (255), 1)
        # cv2.drawContours(mask, [box], 0, (255), 1)

    return b


def recognize(img):
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)


    for (i, c) in enumerate(cnts[0:1]):
        epsilon = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        approx_area = cv2.contourArea(approx)
        print(epsilon, approx_area)
        cv2.drawContours(mask, [approx], -1, (255), 1)
        hull = cv2.convexHull(approx)
        #cv2.drawContours(mask, [hull], -1, (255), 3)
        hullArea = cv2.contourArea(hull)
        print(approx_area)
        if hullArea == 0:
            print("dupa")
            solidity = 0
        else:
            solidity = approx_area / float(hullArea)
        char = "-"

        # if the solidity is high, then we are examining an `O`
        if solidity > 0.9:
            char = "O"

        # otherwise, if the solidity it still reasonabably high, we
        # are examining an `X`
        elif solidity > 0.2:  # or solidity <0.1:
            char = "X"

        # if the character is not unknown, draw it
        if 1:#char != "?":
            cv2.drawContours(mask, [c], -1, (255, 0, 0), 3)
            cv2.putText(mask, char, (int(mask.shape[1]/2), int(mask.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                        (255, 0, 0), 4)

        # show the contour properties
        print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))
        #cv2.drawContours(mask, [c], -1, (255), 3)
    return mask



def extract(img):

    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)
    for (i, c) in enumerate(cnts[1:]):
        print(cv2.contourArea(c))
        cv2.drawContours(mask, [c], 0, (255), 1)
    #print(get_central_square(img))
    #cv2.drawContours(mask, [get_central_square(img)], 0, (255), 1)
    box = get_central_square(img)
    #cv2.drawContours(mask, [get_central_square(img)[0]], 0, (255), 1)
    print(box)
    #gray[int(0.9 * y):y + int(h * 1.5), int(0.9 * x):int(x + 1.4 * w)]
    if box == -1:
        return [mask]
    img_crop, img_rot = crop_rect(img, box[1])
    boxes = []
    boxes.append(((box[1][0][0], box[1][0][1]), (box[1][1][0], box[1][1][1]), box[1][2]))
    img_crop, img_rot = crop_rect(img, boxes[0])
    bo = cv2.boxPoints(boxes[0])
    bo = np.int0(bo)
    #cv2.drawContours(mask, [bo], 0, (255), 1)
    #img_crop, img_rot = crop_rect(mask, boxes[0])
    box = get_central_square(img_rot)
    bo = cv2.boxPoints(box[1])
    bo = np.int0(bo)
    #cv2.drawContours(img_rot, [bo], 0, (127), 1)
    print(box[1][1][1])
    #return mask
    ret = []
    d =1
    #img_rot=cv2.dilate(img_rot, (3,3))
    cv2.imshow("outt", img_rot)
    ret.append(img_rot[int(1.05*(min(bo[:, 1]))): int(0.95*(max(bo[:, 1])) ),
               int(1.05*(min(bo[:, 0]))):int(0.95*(max(bo[:, 0])))])
    ret.append(img_rot[0:max(bo[:, 1])-d*int(box[1][1][0]), min(bo[:,0]):max(bo[:, 0])])
    ret.append(img_rot[min(bo[:, 1]) +d*int(box[1][1][0]):img_rot.shape[0],
               min(bo[:, 0]):max(bo[:, 0])])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               0:min(img_rot.shape[0],max(bo[:, 0])-d*int(box[1][1][1]))])
    ret.append(img_rot[0:max(bo[:, 1]) - d * int(box[1][1][0]),
               0:min(img_rot.shape[0],max(bo[:, 0])-d*int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]) +d * int(box[1][1][0]):img_rot.shape[0],
               0:min(img_rot.shape[0],max(bo[:, 0])-d*int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               max(0, min(bo[:, 0])+d*int(box[1][1][1])):img_rot.shape[1]])
    ret.append(img_rot[0:max(bo[:, 1]) - d * int(box[1][1][0]),
               max(0, min(bo[:, 0])+d*int(box[1][1][1])):img_rot.shape[1]])
    ret.append(img_rot[min(bo[:, 1]) + d * int(box[1][1][0]):img_rot.shape[0],
               max(0, min(bo[:, 0])+d*int(box[1][1][1])):img_rot.shape[1]])
    r = []

    for i in range(0, len(ret)):
        ret[i] = recognize(ret[i])
        r.append(ret[i])

    return ret
    #return [img_rot]

def linear_coef(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)

def translate(box, w, h):
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

        boards.append(gray[int(0.9*y):y + int(h*1.1), int(0.9*x):int(x + 1.1*w)])
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
    d = 0
    for j in extract(i):
        cv2.imshow(str(t)+"_"+str(d), j)

        d+=1
    #cv2.imshow(str(t)+"s", get_central_square(i))
    t +=1

gray = cv2.cvtColor(iq, cv2.COLOR_BGR2GRAY)
gray = 255 -gray
gray= cv2.Canny(gray, 100, 200)
cnts = cv2.findContours(gray.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(iq, cnts, -1, (255), 3)
cv2.imshow("ou", iq)
cv2.waitKey(0)


"""    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]) ,
               min(bo[:, 0]):max(bo[:, 0])])
    ret.append(img_rot[min(bo[:,1])-d*int(box[1][1][0]):max(bo[:, 1])-d*int(box[1][1][0]), min(bo[:,0]):max(bo[:, 0])])
    ret.append(img_rot[min(bo[:, 1]) +d*int(box[1][1][0]):max(bo[:, 1]) + d * int(box[1][1][0]),
               min(bo[:, 0]):max(bo[:, 0])])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               max(0, min(bo[:, 0])-d*int(box[1][1][1])):min(img_rot.shape[0],max(bo[:, 0])-d*int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]) - d * int(box[1][1][0]):max(bo[:, 1]) - d * int(box[1][1][0]),
               max(0, min(bo[:, 0])-d*int(box[1][1][1])):min(img_rot.shape[0],max(bo[:, 0])-d*int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]) +d * int(box[1][1][0]):max(bo[:, 1]) + d * int(box[1][1][0]),
               max(0, min(bo[:, 0])-d*int(box[1][1][1])):min(img_rot.shape[0],max(bo[:, 0])-d*int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               max(0, min(bo[:, 0])+d*int(box[1][1][1])):min(img_rot.shape[0],max(bo[:, 0])+d*int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]) - d * int(box[1][1][0]):max(bo[:, 1]) - d * int(box[1][1][0]),
               max(0, min(bo[:, 0])+d*int(box[1][1][1])):min(img_rot.shape[0],max(bo[:, 0])+d*int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]) + d * int(box[1][1][0]):max(bo[:, 1]) + d * int(box[1][1][0]),
               max(0, min(bo[:, 0])+d*int(box[1][1][1])):min(img_rot.shape[0],max(bo[:, 0])+d*int(box[1][1][1]))])"""