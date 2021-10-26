import cv2
import numpy as np

def empty(x):
    pass

### Getting the HSV Values of the cloak to be used

# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 400, 400)
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 0, 255, empty)


cap =cv2.VideoCapture(0)
cap.set(10,150)

while True:
    cv2.waitKey(2000)
    ret, initial_frame = cap.read()
    if(ret):
        break

initial_frame = cv2.resize(initial_frame,(400,400))

while True:
    _,img = cap.read()
    img = cv2.resize(img,(400,400))
    
    ##### Updating the HSV Values using trackbars
    
    # canny = cv2.Canny(gray, 100,200,apertureSize=3)
    # h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    # h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    # s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    # s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    # v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    # v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernal = np.ones((3,3),np.uint8)
    lower = np.array([94,92,0])
    upper = np.array([179,255,255])
    mask = cv2.inRange(imgHSV, lower, upper)
    mask = cv2.medianBlur(mask,3)
    mask = cv2.dilate(mask,kernal,3)
    mask_inv = 255 - mask

    img_part1 = cv2.bitwise_and(img,img,mask=mask_inv)
    cv2.imshow("IMGPART1",img_part1)

    img_part2 = cv2.bitwise_and(initial_frame,initial_frame,mask = mask)
    cv2.imshow("IMGPART2",img_part2)

    final = cv2.bitwise_or(img_part1,img_part2)
    final = cv2.resize(final,(600,600))
    cv2.imshow("Final",final)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


cv2.destroyAllWindows()
