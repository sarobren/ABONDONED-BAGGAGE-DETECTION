from collections import Counter, defaultdict

import cv2
import numpy as np

file_path ="D:\\New folder\\Project.mp4"
cap = cv2.VideoCapture(file_path)
counter=0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break
    cv2.imwrite('kang'+str(counter)+'.jpeg',frame)
    counter = counter + 1
    
    
    cv2.imshow('Window',frame)
    
     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print ("Frame Count   : ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print ("Format        : ", cap.get(cv2.CAP_PROP_FORMAT))
print ("Height        : ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print ("Width         : ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print ("Mode          : ", cap.get(cv2.CAP_PROP_MODE))
print ("Brightness    : ", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print ("Fourcc        : ", cap.get(cv2.CAP_PROP_FOURCC))
print ("Contrast      : ", cap.get(cv2.CAP_PROP_CONTRAST))
print ("FrameperSec   : ", cap.get(cv2.CAP_PROP_FPS))
FrameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
FrameperSecond =cap.get(cv2.CAP_PROP_FPS)
DurationSecond = FrameCount/FrameperSecond
if FrameperSecond>0:
    print ("FrameDuration : ", DurationSecond, "seconds")
cap.release()
cv2.destroyAllWindows()
pathfile ="C:\\Users\\bibha\\kang148.jpeg"
cv2.namedWindow('CannyEdgeDet',cv2.WINDOW_NORMAL) 
cv2.namedWindow('frame',cv2.WINDOW_NORMAL) 
cv2.namedWindow('Abandoned Object Detection',cv2.WINDOW_NORMAL) 
cv2.namedWindow('frame_masked',cv2.WINDOW_NORMAL)
cv2.namedWindow('Morph_Close',cv2.WINDOW_NORMAL) 
img_original = cv2.imread(pathfile)
cv2.imshow('original', img_original)
mask = np.zeros(img_original.shape[:2], dtype = "uint8")  
pts = np.array([[280,80],[0,300],[0,500],[650,500],[650,80]], np.int32)
cv2.fillPoly(mask,[pts],255,1)
cv2.imshow('Masked',mask)
file_path ="D:\\New folder\\Project.mp4"
cap = cv2.VideoCapture(file_path)
fgbg = cv2.createBackgroundSubtractorMOG2()
consecutiveframe=20
track_temp=[]
track_master=[]
track_temp2=[]
top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)
frameno = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==0:
        break
    frameno = frameno + 1
    cv2.putText(frame,'%s%.f'%('Frameno:',frameno), (400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    fgmask = fgbg.apply(frame)
    fgmask_masked = cv2.bitwise_and(fgmask,fgmask,mask=mask)
    edged = cv2.Canny(fgmask_masked,30,100) 
    cv2.imshow('CannyEdgeDet',edged)
    kernel2 = np.ones((10,10),np.uint8) 
    thresh2 = cv2.morphologyEx(fgmask_masked,cv2.MORPH_CLOSE, kernel2,iterations=2)
    cv2.imshow('Morph_Close', thresh2)  
    (cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mycnts =[] #
    for c in cnts:
       
        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        
            if cv2.contourArea(c) < 100 or cv2.contourArea(c)>20000:
                pass
            else:
                mycnts.append(c)                  
    
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
         
                cv2.putText(frame,'C %s,%s,%.0f'%(cx,cy,cx+cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)                 
    
                sumcxcy=cx+cy            
                          
                track_temp.append([cx+cy,frameno])
                              
                track_master.append([cx+cy,frameno])
                countuniqueframe = set(j for i, j in track_master) 
                      
                
                if len(countuniqueframe)>consecutiveframe: 
                    minframeno=min(j for i, j in track_master)
                    for i, j in track_master:
                        if j != minframeno: 
                            track_temp2.append([i,j])
                
                    track_master=list(track_temp2)
                    track_temp2=[]
                                                          
                countcxcy = Counter(i for i, j in track_master)
              
                for i,j in countcxcy.items(): 
                    if j>=consecutiveframe:
                        top_contour_dict[i] += 1
  
                
                if sumcxcy in top_contour_dict:
                    if top_contour_dict[sumcxcy]>100:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        cv2.putText(frame,'%s'%('CheckObject'), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                        print ('Detected : ', sumcxcy,frameno, obj_detected_dict)
                        
                        obj_detected_dict[sumcxcy]=frameno
    for i, j in obj_detected_dict.items():
        if frameno - obj_detected_dict[i]>200:
            print ('PopBefore',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopBefore : top_contour :',top_contour_dict)
            obj_detected_dict.pop(i)
            
            top_contour_dict[i]=0
            print ('PopAfter',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopAfter : top_contour :',top_contour_dict)

    cv2.polylines(frame,[pts],True,(255,0,0),thickness=2)
    cv2.imshow('Abandoned Object Detection',frame)
    cv2.imshow('frame',fgmask)
    cv2.imshow('frame_masked',fgmask_masked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
top_contours = sorted(top_contour_dict,key=top_contour_dict.get,reverse=True) 
for i in top_contours:
    print (i, top_contour_dict[i]) 
print ("Contours recorded :",len(top_contours))
for i, j in top_contour_dict.items():
    print (i, j)
print ("--")
for i, j in obj_detected_dict.items():
    print (i, j)
track_master = [[781, 153], [738, 153], [662, 153], [780, 154], [720, 154], [781, 155], [725, 155], [779, 156], [733, 156]]
print (track_master)
countcxcy = Counter(i for i, j in track_master)
print (countcxcy)
type(countcxcy)
for i,j in countcxcy.items():
    print (i,j)
mydict1=dict()
mydict1['A']=1
print (mydict1)
mydict1['A'] += 1
print (mydict1)
fileout_path="E:\\project\\OUTPUT\\temp.txt"
fileout = open(fileout_path,'w')
for cxcy,frameno in track_temp:
    fileout.write('%s;%s\n'%(cxcy,frameno))
fileout.close()
mainlist=[[100,1],[200,1],[103,2],[100,2],[105,2],[100,3],[109,3],[501,3]]
x = min(j for i,j in mainlist)
print (x)
y = max(i for i,j in mainlist)
print (y)
unique = set(j for i,j in mainlist)
print (unique)
print (len(unique))
all1 = [1,1,2,2,3,3,3,3]
unique = set(all1)
print (unique,len(unique))
mainlist=[[100,1],[200,1],[103,2],[100,2],[105,2],[100,3],[109,3],[501,3]]
newlist=[]
print (mainlist)
print (len(mainlist))
print (mainlist[0])
print ("mainlist clear :", mainlist)
for i,j in mainlist:
    print (i,j)
    if j != 3: 
        newlist.append([i,j])
print ('mainlist', mainlist)
print ('newlist',newlist)

print ('transfer to mainlist')
mainlist=list(newlist)
print ('updated mainlist', mainlist)
mainlist.append([100,10])
print ('mainlist with new items ', mainlist)
print (newlist)
from collections import Counter

mylist = [1,2,3,3,3,3,5,4,5,6,7,7,7,8,8,8,9]
data1 = Counter(mylist)
print (data1,len(mylist))
data3=data1.most_common(3)
print (data3)
for (i,j) in data3:
    print (i)
mylist = [1,2,3,3,3,3,5,4,5,6,7,7,7,8,8,8,9]
from collections import defaultdict

d = defaultdict(int)
for i in mylist:
    d[i] +=1
    print (d,len(d))
file_path ="D:\\New folder\\Project.mp4"
firstframe_path ="C:\\Users\\bibha\\kang0.jpeg"
firstframe = cv2.imread(firstframe_path)
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray,(21,21),0)
cv2.imshow('Firstframe_blur',firstframe_blur)
cap = cv2.VideoCapture(file_path)
counter = 0
frameno =0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret==0:
        break
    frameno = frameno + 1
    cv2.line(frame,(100,150),(100,400),(0,255,0),2)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)
    cv2.imshow('frame_blur',frame_blur)
    frame_difference = cv2.absdiff(firstframe_blur, frame_blur)
    cv2.imshow('frame_diff',frame_difference)
    edged = cv2.Canny(frame_difference,30,100) 
    cv2.imshow('Edge',edged)
    kernel = np.ones((15,15),np.uint8) #
    thresh1 = cv2.dilate(edged, kernel, iterations=3)
    cv2.imshow('Dilate',thresh1)
    kernel2 = np.ones((20,20),np.uint8) 
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel2,iterations=2)
    cv2.imshow('Morph_Close', thresh2)  
    (cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mycnts =[] 
    for c in cnts:
        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
      
            if cv2.contourArea(c) < 200 or cv2.contourArea(c)>20000:
                pass
            elif (cv2.contourArea(c)>5000 and cx<70):
                pass
            else:
                mycnts.append(c)
           
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
          
                cv2.putText(frame,'C %s,%s'%(cx,cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
                
                print (cv2.contourArea(c), cx, cy, frameno)
                     
                cv2.putText(frame,'%s%s'%('Objects :',len(mycnts)), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.imshow('Window',frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
mcx=[0,0]
print (mcx)
mcx=[0,138]
print (mcx)
print (mcx[0]>60 and mcx[1]<60)
mcx.pop(0)
print (mcx)
mcx.append(109)
print (mcx)
mcx.pop(0)
print (mcx)
file_path ="D:\\New folder\\Project.mp4"
firstframe_path ="C:\\Users\\bibha\\kang0.jpeg"

firstframe = cv2.imread(firstframe_path)
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray,(21,21),0)
cv2.imshow('Firstframe_blur',firstframe_blur)
cap = cv2.VideoCapture(file_path)
counter = 0
frameno =0
comparecx=[0] 
sumarea=0
PaxbyArea=0
Height = 200
Width = 200
while (cap.isOpened()):
    ret, frame = cap.read()
 
    if ret==0:
        break
    frameno = frameno + 1
    cv2.line(frame,(100,150),(100,400),(0,255,0),2)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)
    cv2.imshow('frame_blur',frame_blur)
    frame_difference = cv2.absdiff(firstframe_blur, frame_blur)
    cv2.imshow('frame_diff',frame_difference)
    edged = cv2.Canny(frame_difference,30,50) 
    cv2.imshow('CannyEdgeDetection',edged)
    kernel = np.ones((5,5),np.uint8) 
    thresh = cv2.dilate(edged, kernel, iterations=3)
    cv2.imshow('Dilate',thresh)
    kernel2 = np.ones((8,8),np.uint8) 
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel2,iterations=3)
    cv2.imshow('Morph_Close', thresh2)  
   
    (cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       
    cv2.putText(frame,'%s'%('l'), (100,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    for i, c in enumerate(cnts):
       
        M = cv2.moments(cnts[i])
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
               
        if cv2.contourArea(c) > 2000 and cx<150 and cy>150:
     
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,'C %s,%s'%(cx,cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
 
            print (cv2.contourArea(c), cx, cy, frameno, "-")
                      
            comparecx.append(cx)
            if comparecx[0]>100 and comparecx[1]<100 : 
                counter = counter + 1
                sumarea = sumarea + cv2.contourArea(c) 
                PaxbyArea = sumarea/6500 
                print ("Pax ",counter, " - ",cv2.contourArea(c))
            comparecx.pop(0)
             
    cv2.putText(frame,'%s'%('ByCountline:'), (10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    cv2.putText(frame,'%s'%(counter), (10,80),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    cv2.putText(frame,'%s'%('ByContourArea:'), (350,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    cv2.putText(frame,'%.2f'%(PaxbyArea), (350,80),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    cv2.imshow('Window',frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
file_path ="D:\\New folder\\Project.mp4"
cap = cv2.VideoCapture(file_path)
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    try:
        cv2.imshow('frame',fgmask)
    except:
        pass
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
file_path ="D:\\New folder\\Project.mp4"
cap = cv2.VideoCapture(file_path)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorGMG2()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('frame',fgmask)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()