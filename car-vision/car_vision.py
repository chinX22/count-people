import cv2
import numpy as np
def find_lines(img):
    img2 = img[int(img.shape[0] * 0.65): img.shape[0]]
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("half", gimg2)s
    #cv2.waitKey(0)
    edges = cv2.Canny(gray, 150, 150,apertureSize = 3)
    #return edges
    #g1 = cv2.GaussianBlur(gray, (1,1), 0);
    #g2 = cv2.GaussianBlur(gray, (9,9), 0);
    #edges = g1 - g2;
    #cv2.imshow("edges", edges)
   # cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges,rho = 1, theta = np.pi/180,threshold = 60,
                           minLineLength = 40, maxLineGap = 10)
    try:
        minDiff = 10
        for i in range(len(lines)):
            x1,y1,x2,y2 = lines[i][0]
            if (abs(x1-x2) < minDiff or abs(y1-y2) < minDiff):
                continue
            cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
        
    except:
        print("raise")
        raise
    return img2
    #cv2.imshow("find lines", img2)
    #cv2.waitKey(0)
    
img = cv2.imread('road_ss.png')
cv2.imshow("Output", find_lines(img))
cv2.waitKey(0)

cap = cv2.VideoCapture("driving_clip.mp4")
while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
                break
        cv2.imshow("Output", find_lines(frame))
        if cv2.waitKey(1) == ord('q'):
                break
cap.release()