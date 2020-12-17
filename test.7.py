import cv2
import imutils
import statistics
import numpy as np
import math
from decimal import Decimal

image = cv2.imread("a8.jpg")
kernel = np.ones((3,3), np.uint8)
#image = cv2.resize(image, (1000, 750))
#image = cv2.dilate(image, kernel, iterations=1)
iq = image.copy()

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    angle = rect

    if angle < 90:
        angle = (90-angle)
    else:
        angle = angle-90
    print("kat", 90-angle)


    return img, imutils.rotate_bound(img, angle)


def get_central_square(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1]), reverse=True)

    for (i,c) in enumerate(cnts[0:1]):
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        hull = cv2.convexHull(approx)
    #print(hull)
    hull1 = np.ndarray((8,1,2), dtype=np.int32)
    if len(hull) > 8:
        ind = 0
        ind1 = 0
        for i in hull:
            add = True
            for j in hull[ind+1:]:

                if (abs(i[0][0] - j[0][0]) < 20) and (abs(i[0][1] - j[0][1]) < 20) and ((i[0][0] != j[0][0]) \
                        or (i[0][1] != j[0][1])) :
                    #print(i, j)
                    add = False
            if add == True:
                hull1[ind1][0][0] = np.int32(i[0][0])
                hull1[ind1][0][1] = np.int32(i[0][1])
                ind1+=1
            ind +=1
        return hull1
    #print(hull.dtype)
    #print(hull1)

    return hull


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

def get_angle(hull):
    print(hull)
    print("dupa")
    y_max = hull[hull[:,0,0].argsort()[::-1][:2]]
    y_min = hull[hull[:, 0, 0].argsort()[:2]]
    x_max = hull[hull[:, 0, 1].argsort()[::-1][:2]]
    x_min = hull[hull[:, 0, 1].argsort()[:2]]
    y_max = y_max[y_max[:, 0, 1].argsort()]
    y_min = y_min[y_min[:, 0, 1].argsort()]
    x_max =x_max[x_max[:, 0, 0].argsort()]
    x_min =x_min[x_min[:, 0, 0].argsort()]
    print('y_max',y_max)
    a = np.arctan((y_max[1][0][0] - y_min[1][0][0])/(y_max[1][0][1] - y_min[1][0][1]))
    a = np.degrees(abs(a))
    tmp = np.arctan((y_max[0][0][0] - y_min[0][0][0]) / (y_max[0][0][1] - y_min[0][0][1]))
    tmp = np.degrees(abs(tmp))
    a = (a+tmp)/2


    print(a)
    return a

def points(hull):
    y_max = hull[hull[:,0,0].argsort()[::-1][:2]]
    y_min = hull[hull[:, 0, 0].argsort()[:2]]
    x_max = hull[hull[:, 0, 1].argsort()[::-1][:2]]

    x_min = hull[hull[:, 0, 1].argsort()[:2]]
    y_max = y_max[y_max[:, 0, 1].argsort()]
    y_min = y_min[y_min[:, 0, 1].argsort()]
    x_max =x_max[x_max[:, 0, 0].argsort()]
    x_min =x_min[x_min[:, 0, 0].argsort()]
    print(y_max)
    a1 = ((y_max[0][0][0] - y_min[0][0][0])) / ((y_max[0][0][1] - y_min[0][0][1]))
    b1 = (y_min[0][0][1]- (a1 * y_min[0][0][0]))
    print(a1, b1)
    p1 = line_intersection((y_max[0][0], y_min[0][0]), (x_min[0][0], x_max[0][0]))
    p2 =line_intersection((y_max[0][0], y_min[0][0]), (x_min[1][0], x_max[1][0]))
    p3 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[0][0], x_max[0][0]))
    p4 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[1][0], x_max[1][0]))

    bo = np.array([p1, p2, p3, p4], dtype=np.int)
    for i in bo:
        i[0] = int(i[0])
        i[1] = int(i[1])

    print(bo)
    #print("min", bo[:, 1])
    return bo
def w_h(hull):
    y_max = hull[hull[:,0,0].argsort()[::-1][:2]]
    y_min = hull[hull[:, 0, 0].argsort()[:2]]
    x_max = hull[hull[:, 0, 1].argsort()[::-1][:2]]

    x_min = hull[hull[:, 0, 1].argsort()[:2]]
    y_max = y_max[y_max[:, 0, 1].argsort()]
    y_min = y_min[y_min[:, 0, 1].argsort()]
    x_max =x_max[x_max[:, 0, 0].argsort()]
    x_min =x_min[x_min[:, 0, 0].argsort()]
    print(y_max)
    a1 = ((y_max[0][0][0] - y_min[0][0][0])) / ((y_max[0][0][1] - y_min[0][0][1]))
    b1 = (y_min[0][0][1]- (a1 * y_min[0][0][0]))
    print(a1, b1)
    p1 = line_intersection((y_max[0][0], y_min[0][0]), (x_min[0][0], x_max[0][0]))
    p2 =line_intersection((y_max[0][0], y_min[0][0]), (x_min[1][0], x_max[1][0]))
    p3 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[0][0], x_max[0][0]))
    p4 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[1][0], x_max[1][0]))
    bo = np.array([p1, p2, p3, p4])
    h = (len_between_points(p1, p2) + len_between_points(p4, p3))/2
    w =(len_between_points(p1, p3) + len_between_points(p2, p4))/2
    #print("min", bo[:, 1])
    box = [[0],[[0],[w,h]]]
    print(box[1][1][1])
    return box

def len_between_points(p1, p2):
    print("p1", p1)
    x = (p1[0] - p2[0])*(p1[0] - p2[0])
    y = (p1[1] - p2[1])*(p1[1] - p2[1])
    a = math.sqrt(x+y)
    return a

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]



def extract(img):

    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)
    for (i, c) in enumerate(cnts[0:1]):
        print(cv2.contourArea(c))
        cv2.drawContours(mask, [c], 0, (255), 1)
    #print(get_central_square(img))
    #cv2.drawContours(mask, [get_central_square(img)], 0, (255), 1)
    box = get_central_square(img)
    a = get_angle(box)
    img_crop, img_rot = crop_rect(img,a)
    box = get_central_square(img_rot)
    bo = points(box)
    ret = []
    b = box
    box = w_h(box)
    d = 1
    print('min',min(bo[:, 1]))
    ret.append(img_rot[int(1.05 * (min(bo[:, 1]))): int(0.95 * (max(bo[:, 1]))),
               int(1.05 * (min(bo[:, 0]))):int(0.95 * (max(bo[:, 0])))])

    ret.append(img_rot[0:int(max(bo[:, 1]) - 1 * int(box[1][1][0])), int(min(bo[:, 0])):int(max(bo[:, 0]))])
    ret.append(img_rot[min(bo[:, 1]) + d * int(box[1][1][0]):img_rot.shape[0],
               min(bo[:, 0]):max(bo[:, 0])])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               0:min(img_rot.shape[0], max(bo[:, 0]) - d * int(box[1][1][1]))])
    ret.append(img_rot[0:max(bo[:, 1]) - d * int(box[1][1][0]),
               0:min(img_rot.shape[0], max(bo[:, 0]) - d * int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]) + d * int(box[1][1][0]):img_rot.shape[0],
               0:min(img_rot.shape[0], max(bo[:, 0]) - d * int(box[1][1][1]))])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               max(0, min(bo[:, 0]) + d * int(box[1][1][1])):img_rot.shape[1]])
    ret.append(img_rot[0:max(bo[:, 1]) - d * int(box[1][1][0]),
               max(0, min(bo[:, 0]) + d * int(box[1][1][1])):img_rot.shape[1]])
    ret.append(img_rot[min(bo[:, 1]) + d * int(box[1][1][0]):img_rot.shape[0],
               max(0, min(bo[:, 0]) + d * int(box[1][1][1])):img_rot.shape[1]])


    print(len(box))
    cv2.drawContours(mask, [b], 0, (255), 1)
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

        boards.append(gray[int(0.8*y):y + int(h*1.2), int(0.8*x):int(x + 1.2*w)])
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