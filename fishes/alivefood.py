import numpy as np
import cv2
import random as rand
import math
from math import sin,cos,pi
from utils import get_subwindow

initial_energy = 20
max_speed = 7
min_speed = 1
minsight = 14

perc_square_edge = 5

class AliveFood():
    def __init__(self,pos=(300,300),orient=pi*0,max_radius = 30,mind=None):
        self.area = np.zeros((max_radius*2,max_radius*2,3),dtype=np.uint8)

        s1,s2 = np.array(self.area.shape[:2])/2


        self.s1 = int(s1)
        self.s2 = int(s2)
        #self.color = [np.asscalar(e) for e in color]
        self.r = int(max_radius)
        self.lifetime=0
        
        #cv2.circle(self.area,(self.s1,self.s2),1,self.color,5)

        self.pos=pos


        self.speed = rand.randint(min_speed, max_speed)
        self.orient=orient
        self.mind = self.init_mind(variables = mind)
        #fitness value
        self.diversity_speed=np.zeros(3)
        self.diversity_turn=np.zeros(3)


    def init_mind(self,visible=8,h_n=5,output = 2,variables=None):
        if variables is None:
            variables = np.random.rand(h_n*(visible+1)+ (h_n+1)*output)*2 -1
        #print(variables.shape,h_n*(visible+1)+ (h_n+1)*output)
        h_l = variables[0:h_n*(visible+1)].reshape(visible+1,h_n)#3 input 2 output
        out_l =variables[h_n*(visible+1):].reshape(h_n+1,output)#3 input 1 outout
        #print(h_l.shape)
        #print(out_l.shape)
        self.predict = lambda p: np.tanh( #final activation for every neuron
                np.dot(out_l.T,
                    np.hstack((#adding the bias for the hidden
                        np.tanh(#output of hidden
                            np.dot(h_l.T, #hidden
                                    p
                                    )),np.ones(1)))))
                      
        self.mind = variables
        return variables

    def draw(self,world):
        s1 = self.s1
        s2 = self.s2
        p1,p2  = map(int,self.pos)
        
        center=(self.s1,self.s2)#center  = tuple(int(max_radius) * np.ones(2,dtype=np.uint8))
        max_radius = self.r
        self.area = 0*np.ones((max_radius*2,max_radius*2,3),dtype=np.uint8)
            #print(food_img.shape)
        d1 = d2 = int(max_radius)
        for i in range(max_radius,5,-2):
                #255-int(i/radius * 255)

            color =np.array((255,255,255))-(i/max_radius *255)
                #color = (255,0,0)
                #print(i,color)
            cv2.circle(self.area,center,i,color,1)
                #cv2.imshow('circle',food_img)#merge_images_h([food_img]))
                #cv2.waitKey(0)
            
    #            p1,p2 = self.pos[::-1]
        
        world[p1-d1:p1+d1,p2-d2:p2+d2] = np.minimum(world[p1-d1:p1+d1,p2-d2:p2+d2]-self.area.astype(np.uint8),world[p1-d1:p1+d1,p2-d2:p2+d2])
        cv2.circle(world,(p2,p1),5,(0,1,1),-1)
        

    def print_perception(self,world):
        
        s1 = minsight #sight distance
        s2 = minsight # sight distance
        off1,off2  = map(int,self.pos)
        for i in range(0,8):
            angle = (i/4)*pi-self.orient
            x1,x2 = (int(s1*cos(angle)),int(s2*sin(angle)))
            cv2.rectangle(world, 
                            (off2+x1-perc_square_edge-1,off1+ x2-perc_square_edge-1),
                             (off2+x1+perc_square_edge,off1+ x2+perc_square_edge),
                            
                             (0,255,40*i))
    def get_perception(self,world):
        s1 =minsight
        s2 =minsight
        off1,off2 = map(int,self.pos[::-1])
        
        inputs = []
        real_world = np.array(world)
        for i in range(0,8):
            angle = (i/4)*pi-self.orient
            x1,x2 = (int(s1*cos(angle)),int(s2*sin(angle)))
            #print('they sould be around<<50',off1+x1,dim1,off2+x2,dim2)
            perc = get_subwindow(real_world,
                                    (off1+x2,off2+x1),
                                    deltax=perc_square_edge,
                                    deltay=perc_square_edge)
            
            #cv2.rectangle(world, (off1+x1-dim1-1,off2+ x2-dim2-1), (off1+x1+dim1,off2+ x2+dim2),(0,0,255))
            height, width = perc.shape[:2]
            
            #print(height,width,(off1+x2,off2+x1))
            res = cv2.resize(perc,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
            inputs.append(res)
        self.percs=inputs
        #if np.array(inputs).shape != (7,30,30,3):
            #print(np.array(inputs))
            #print(np.array(inputs).shape)
        self.to_feed = np.hstack([ np.mean( #the 8 inputs
                            np.max(np.array(inputs)!=(255,255,255),axis=3)#compare channels to somethig
                                ,axis=(1,2)),#the mean of the square
                                #self.speed,self.orient,
                                1]#bias
                        )#.reshape(8,1)#reshaping
        #print(len(inputs))
        return inputs
    def move(self,world,dx=25,dy=25):
        #print(np.array(speed*np.array([math.cos(self.orient),math.sin(self.orient)]),dtype=int))
        speed = self.speed
        #print(speed)
        self.pos += np.array(speed*np.array([math.cos(self.orient),
                        math.sin(self.orient)]),dtype=int)
        s1 = self.s1
        s2 = self.s2
        p1,p2 = self.pos
        
        #sforo sopra
        if p1-s1<dx:
            p1 = world.shape[0]-dx-s1#s1+dx
        #sforo sotto
        elif p1+s1>world.shape[0]-dx:
            p1 = s1+dx#world.shape[0]-s1-dx
        #sforo sinistra
        if p2-s2< dy :
            p2=world.shape[1]-dy-s2#s2+dy
        #sforo destra
        elif p2+s2>world.shape[1]-dy:
            p2 = s2+dy#world.shape[1]-s2-dy
        self.pos = (p1,p2)
        self.lifetime += 1
        #print(round(self.orient*180 / pi %360,0))

    def inc_speed(self,delta):
        new_speed = self.speed + delta
        new_speed = new_speed - (new_speed%max_speed)*(new_speed>=max_speed)
        self.speed = new_speed * (new_speed>=min_speed) + min_speed*(new_speed<min_speed)

    def turn(self,direction=-1):

        self.orient = direction*pi
        self.orient  = (self.orient*180 / pi %360)*pi/180


