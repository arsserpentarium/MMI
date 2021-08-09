# This is python script, which generate random numbers from noises in camera sensor.
# Core principle is simple: it encodes color values of every frame in gray code, compares values of two consequent
# frames bit by bit. And writes to output stream all bits, which changed between frames. This version of script uses
# only 4 lesser bits, because higher bits doesn't change so often. So, without them script works much faster.
# That algorithm is based on https://github.com/AndieNoir one. It just lose less entropy in process.

def maskbit(a,b):
    r=np.array([],dtype=np.bool_)
    #a= u ^ (u>>1)
    #b= v ^ (v>>1)
    x = a ^ b
    for i in range(0,8):
        if 1&(x>>i):
            if 1&(a>>i):
                r = np.append(r, [True])
            else:
                r = np.append(r, [False])
    return r

import numpy as np
import bitstream as bs
import cv2
import time

cap = cv2.VideoCapture(0)
a=bs.BitStream()
fl=-1
ret, frame = cap.read()
print(frame.shape)
frame2=frame

lut=np.empty(256,dtype=np.dtype(object))
for x in range(0,256):
    lut[x] =maskbit(x&15,x>>4)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if np.array_equal(frame,frame2)==False:
        fr=frame&15
        if (fl==1):
            a.write(np.concatenate(lut.take(np.ravel(fr|(fr2<<4)))),bool)
            l=a.__len__()
            print(a.__len__())


        else:
            fr2 = fr
        fl=-fl
        frame2=frame


    # Display the resulting frame

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
file = open(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())+'.bin', "wb")
file.write(a.read(bytes, l//8))
file.close()
cap.release()
cv2.destroyAllWindows()
