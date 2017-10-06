# Detection-and-Translation-of-Sign-Language

# import the necessary packages
import imutils
import cv2
import numpy as np
import math



cap = cv2.VideoCapture(0);
while(cap.isOpened()):
    # load the image, convert it to grayscale, and blur it slightly
    ret, img = cap.read()
    ret2, graynums = cap.read()
    wndw = cv2.cvtColor(graynums, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
 
    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
    
    hull = cv2.convexHull(c)
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    c_area = int(M['m00'])
    perim = cv2.arcLength(c, True)
    totcont = len(c)
    
    cv2.circle(img, (cx,cy), 10, (255,0,255), -1)
    cv2.putText(img, "c(%s, %s)" %(cx,cy), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1) 

# Calculations
    cv2.putText(wndw, "Centroid (%s, %s)" %(cx,cy), (0,50), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)
    cv2.putText(wndw, "North (%s, %s)" %(extTop), (0,80), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)
    cv2.putText(wndw, "East (%s, %s)" %(extRight), (0,110), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)
    cv2.putText(wndw, "West (%s, %s)" %(extLeft), (0,140), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)
    cv2.putText(wndw, "South (%s, %s)" %(extBot), (0,170), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)

##    cv2.putText(wndw, "Total Number of Contours = %i" %(totcont), (0,200), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)
    cv2.putText(wndw, "Contour Area = %.2f" %(c_area), (0,230), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)
##    cv2.putText(wndw, "Contour Perimeter = %.2f" %(perim), (0,260), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 1)


    cv2.circle(img, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(img, extRight, 8, (0, 255, 0), -1)
    cv2.circle(img, extTop, 8, (255, 0, 0), -1)
    cv2.circle(img, extBot, 8, (255, 255, 0), -1)

##    cv2.putText(extLeft, "West: {}", (10, img.shape[0] - 10), 0.35, (0,0, 255), 1)
    # show the output image
    cv2.imshow("Image", img)
    cv2.imshow("Le Values", wndw)


    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
