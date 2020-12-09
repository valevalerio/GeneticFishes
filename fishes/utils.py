import numpy as np
import cv2
def merge_images_h(l):
    shapes = np.array([i.shape for i in l])
    maxx = np.max(shapes[:,0]) #la prima
    final = np.zeros((maxx,0,3), dtype=np.uint8)
    
    for i in l:
        copied = np.zeros((maxx,i.shape[1],3),dtype=np.uint8)
        copied[:i.shape[0],:i.shape[1]+2,:]=i.astype(np.uint8)
        
        final=np.hstack((final,copied))
        #final=np.hstack((final,np.zeros((maxx,10,3))))
        final=np.hstack((final,np.zeros((maxx,10,3), dtype=np.uint8)))
    return final

def merge_images_v(l):
    shapes = np.array([i.shape for i in l])
    maxy = np.max(shapes[:,1]) #la seconda
    #print (maxy)
    final = np.zeros((0,maxy,3), dtype=np.uint8)
    for i in l:
        copied = np.zeros((i.shape[0],maxy,3),dtype=np.uint8)
        #print ('copied',i.shape,'in',copied[:i.shape[0],:i.shape[1]].shape)
        copied[:i.shape[0],:i.shape[1]]=i.astype(np.uint8)
        final=np.vstack((final,copied))
        final=np.vstack((final,np.zeros((10,maxy,3), dtype=np.uint8)))
    return final

def command_bridge(size_w1=300,commands=['D for Debug','E for Energy'],f_size=0.7,colors=None):
    if colors is None:
        colors = [(47,79,79) for i in range(len(commands))]
    size_w1 = len(commands)*30+10
    size_bridge = (size_w1,300,3)
    command_bridge = np.zeros(size_bridge)+(169,169,169)
    command_bridge[0:5,:] = (105,105,105)
    command_bridge[-5:,:] = (105,105,105)
    command_bridge[:,0:5] = (105,105,105)
    command_bridge[:,-5:] = (105,105,105)
    for i,(comm,col) in enumerate(zip(commands,colors)):
        cv2.putText(command_bridge, comm ,(10,30*(i)+25), cv2.FONT_HERSHEY_SIMPLEX, f_size,col,2,cv2.LINE_AA)

    return command_bridge

def get_subwindow(arr,pt,deltax=50,deltay=50):
    x,y = int(pt[0]),int(pt[1])
    px = np.max([x-deltax,0])
    pdx = np.min([x+deltax,arr.shape[1]])
    py = np.max([y-deltay,0])
    pdy = np.min([y+deltay,arr.shape[0]])
    sub = arr[py:pdy,px:pdx]
    return sub

