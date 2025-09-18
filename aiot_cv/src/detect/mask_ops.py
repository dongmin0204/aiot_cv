import cv2, numpy as np

def crop_square_roi(img, xyxy, out_size=256, scale=1.2):
	x1,y1,x2,y2 = map(int, xyxy)
	w,h = x2-x1, y2-y1
	s = int(scale*max(w,h)/2)
	cx,cy = (x1+x2)//2, (y1+y2)//2
	x1n,y1n = max(0,cx-s), max(0,cy-s)
	x2n,y2n = min(img.shape[1]-1,cx+s), min(img.shape[0]-1,cy+s)
	roi = img[y1n:y2n, x1n:x2n]
	roi = cv2.resize(roi, (out_size,out_size))
	return roi, {"x1":x1n,"y1":y1n,"size":(x2n-x1n)}
