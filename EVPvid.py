import numpy as np
import cv2
from cv2 import aruco
import time

# That script is a bit more modern approach for common transcommunication method, which is based on video loop.
# That program requires certain setup for work: monitor, camera and mirror. Camera must observe monitor through
# mirror. That script perform video loop with automatic window recognition based on aruco markers.


bordersize1 = 200# size of aruco markers
bordersize2 = 10# size of border indent
borderc = 191# brightness of border
threshold = 128# image recognition threshold
size = 2# magnification of camera window

cap = cv2.VideoCapture(-1)

ret, frame = cap.read()
print(frame.shape)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

fw = frame.shape[1]*size
fh = frame.shape[0]*size

border = np.ones([bordersize1*2+bordersize2*4+fh, bordersize1*2+bordersize2*4+fw, 3], dtype=np.uint8)*borderc
xyo = (bordersize1 + bordersize2*2)
x_offset = np.array([bordersize2, int(bordersize1*0.5+bordersize2*2+fh/2), bordersize1+bordersize2*3+fh, bordersize2, bordersize1+bordersize2*3+fh, bordersize2, int(bordersize1*0.5+bordersize2*2+fh/2), bordersize1+bordersize2*3+fh], dtype=np.uint16)
y_offset = np.array([bordersize2, bordersize2, bordersize2, int(bordersize1*0.5+bordersize2*2+fw/2), int(bordersize1*0.5+bordersize2*2+fw/2), bordersize1+bordersize2*3+fw, bordersize1+bordersize2*3+fw, bordersize1+bordersize2*3+fw], dtype=np.uint16)
g_markers = np.zeros([1, 4, 2], dtype=np.float32)
for i in range(0, 8):
    img = aruco.drawMarker(aruco_dict, i, bordersize1)
    corners = np.array([[[x_offset[i], y_offset[i]], [x_offset[i], y_offset[i]+bordersize1], [x_offset[i]+bordersize1, y_offset[i]+bordersize1], [x_offset[i]+bordersize1, y_offset[i]]]], dtype=np.float32)
    g_markers = np.concatenate((g_markers, corners), axis=0)
    for j in range(0, 3):
        border[x_offset[i]:x_offset[i]+bordersize1, y_offset[i]:y_offset[i]+bordersize1, j] = img

mask1 = np.ones(border.shape, np.uint8)*255
for i in range(8):
    cv2.rectangle(mask1,(y_offset[i], x_offset[i]), (y_offset[i]+bordersize1, x_offset[i]+bordersize1), (0, 0, 0), cv2.FILLED)
cv2.rectangle(mask1, (xyo, xyo), (xyo+fw, xyo+fh), (0, 0, 0), cv2.FILLED)

g_markers = g_markers[1:, :, :]


parameters = aruco.DetectorParameters_create()

out = np.zeros([fh+xyo*2, fw+xyo*2, 3], dtype=np.uint8)
out_r = np.zeros([fh, fw, 3], dtype=np.uint8)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())+'.mp4', fourcc, 20, (width,height))
fl = True
while fl:
    # Capture frame-by-frame
    ret, framer = cap.read()
    frame2 = frame
    frame = cv2.flip(framer, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(thresh, aruco_dict, parameters=parameters)
    if (np.array_equal(frame,frame2)==False) & ~(ids is None) :
        ids2 = np.squeeze(ids)
        ids2 = ids2[ids2 < 8]
        keys = np.argsort(ids2)
        kl = len(keys)
        if (kl > 0) & (kl <= 8):
            corners2 = np.squeeze(np.array(corners)[keys]).reshape((4*kl, 2))
            g_corners = g_markers[ids2[keys], :, :].reshape((4*kl, 2))
            h, status = cv2.findHomography(corners2, g_corners)
            out = cv2.transpose(cv2.warpPerspective(frame, h, (fh+xyo*2, fw+xyo*2), flags=cv2.INTER_CUBIC))

            #mean = (np.array(cv2.mean(out, mask=mask1[:, :, 1]), dtype=float)[:3])/borderc
            #out = (out/mean).clip(0, 255).astype(np.uint8)
            out =out[xyo:xyo+fh, xyo:xyo+fw, :]
            out_r = out
            writer.write(cv2.resize(out_r, (width, height), interpolation=cv2.INTER_AREA))
            thresh = aruco.drawDetectedMarkers(thresh, corners, ids)

    #resized = cv2.resize(aruco.drawDetectedMarkers(frame.copy(), corners, ids), (fw, fh), interpolation=cv2.INTER_AREA)

    border[xyo:xyo+fh, xyo:xyo+fw, :] = out_r

    cv2.imshow('frame', border)
    cv2.imshow('thresh', thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        fl = False
print('end')
# When everything done, release the capture
cap.release()
writer.release()
cv2.destroyAllWindows()
