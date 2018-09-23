import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=10)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()
    fg_mask = fgbg.apply(frame)
    fg__mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Frame', fg_mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
