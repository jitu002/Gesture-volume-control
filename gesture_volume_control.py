import cv2
import mediapipe as mp
import time
import math
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.GetMute()
volume.GetMasterVolumeLevel()
volrng=volume.GetVolumeRange()
volmin=volrng[0]
volmax=volrng[1]
volume.SetMasterVolumeLevel(-20.0, None)

wcam=980
hcam=800
cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpdraw=mp.solutions.drawing_utils
cap.set(3,wcam)
cap.set(4,hcam)
ptime=0

def length(x1,x2,y1,y2):
    h1=x2-x1
    h2=y2-y1
    l=math.sqrt((h1**2+h2**2))
    return l

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    imgrbg=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(frame,f'FPS:-{int(fps)}',(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    result=hands.process(imgrbg)
    if result.multi_hand_landmarks:
        hlist=[]
        for a in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(frame,a,mphands.HAND_CONNECTIONS)
            for id,lm in enumerate(a.landmark):
                h,w,c=frame.shape
                cx=int(lm.x*w)
                cy=int(lm.y*h)
                hlist.append((id,cx,cy))
        if len(hlist)!=0:
            x1,y1=hlist[4][1],hlist[4][2]
            x2,y2=hlist[8][1],hlist[8][2]
            a=(x1+x2)//2
            b=(y1+y2)//2

            cv2.circle(frame,(x1,y1),15,(255,0,0),cv2.FILLED)
            cv2.circle(frame,(x2,y2),15,(255,0,0),cv2.FILLED)
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),3)
            cv2.circle(frame,(a,b),12,(0,255,0),cv2.FILLED)
            lent=int(length(x1,x2,y1,y2))
            vol=np.interp(lent,(40,325),(volmin,volmax))
            volbr=np.interp(lent,(40,325),(310,110))
            volp=np.interp(lent,(40,325),(0,100))
            volume.SetMasterVolumeLevel(vol,None)
            cv2.rectangle(frame,(30,110),(64,310),(255,0,0),3)
            cv2.rectangle(frame,(30,int(volbr)),(64,310),(255,0,0),cv2.FILLED)
            cv2.putText(frame,f'Vol:-{int(volp)}%',(10,350),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            if lent<40:
                cv2.circle(frame,(a,b),12,(0,0,255),cv2.FILLED)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('e'):
        break

cap.release()
cv2.destroyAllWindows()