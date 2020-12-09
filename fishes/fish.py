import numpy as np
import cv2
import random as rand
import math
from math import sin,cos,pi
from utils import get_subwindow

initial_energy = 100 #20
max_speed = 15
min_speed = 3
minsight = 8

perc_square_edge = 5
class Fish():
    def __init__(self,pos=(300,300),orient=pi*0,color=(0,0,255),mind=None):
        self.area = np.zeros((50,50,3),dtype=np.uint8)
        s1,s2 = np.array(self.area.shape[:2])/2
        self.s1 = int(s1)
        self.s2 = int(s2)
        self.color = [np.asscalar(e) for e in color]
        #print(color,type(self.color[0]))
        
        cv2.circle(self.area,(self.s1,self.s2),1,self.color,5)
        self.pos=pos
        self.speed = rand.randint(min_speed, max_speed)
        self.orient= 2*pi*rand.random()
        self.mind = self.init_mind(variables = mind)
        #fitness value
        self.energy=initial_energy
        self.eaten = 0
        self.diversity_speed=np.zeros(3)
        self.diversity_turn=np.zeros(3)


    def init_mind(self,visible=9,h_n=5,output = 2,variables=None):
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
        p1,p2 = self.pos
        
        
        self.area = np.zeros(self.area.shape)#.astype(np.uint8)
        #cv2.circle(self.area,(self.s1,self.s2),self.s1,(255,30,30),1)
        cv2.circle(self.area,(self.s1,self.s2),3,self.color,5)
        tail = (int(s1/3*sin(self.orient+pi))+s1,int(s2/3*cos(self.orient+pi))  +s2 )
        cv2.circle(self.area,tail,2,self.color,5)
        #cv2.line(self.area,(self.s1,self.s2),tail,self.color,3)
        #print(p1,s1,p1-s2,p2,s2,p2-s2,world.shape[:2])
        
        world[p1-s1:p1+s1,p2-s2:p2+s2] += self.area.astype(np.uint8)
        

    def print_perception(self,world):
        
        s1 = minsight + int(self.s1*(self.speed/max_speed))# sight distance
        s2 = minsight + int(self.s2*(self.speed/max_speed)) # sight distance
        off1,off2 = self.pos[::-1]
        for i in range(0,7):
            angle = (i/6)*pi-self.orient
            x1,x2 = (int(s1*cos(angle)),int(s2*sin(angle)))
            cv2.rectangle(world, 
                            (off1+x1-perc_square_edge-1,off2+ x2-perc_square_edge-1),
                             (off1+x1+perc_square_edge,off2+ x2+perc_square_edge),(0,40*i,255))
    def get_perception(self,world):
        s1 =minsight+ int(self.s1*(self.speed/max_speed))# sight distance
        s2 =minsight+ int(self.s2*(self.speed/max_speed)) # sight distance
        off1,off2 = self.pos[::-1]
        inputs = []
        real_world = np.array(world)
        for i in range(0,7):
            angle = (i/6)*pi-self.orient
            x1,x2 = (int(s1*cos(angle)),int(s2*sin(angle)))
            #print('they sould be around<<50',off1+x1,dim1,off2+x2,dim2)
            perc = get_subwindow(real_world,
                                    (off1+x1,off2+x2),
                                    deltax=perc_square_edge,
                                    deltay=perc_square_edge)
            
            #cv2.rectangle(world, (off1+x1-dim1-1,off2+ x2-dim2-1), (off1+x1+dim1,off2+ x2+dim2),(0,0,255))
            height, width = perc.shape[:2]
            #print(perc.shape[:2])
            
            res = cv2.resize(perc,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
            inputs.append(res)
        self.percs=inputs
        #if np.array(inputs).shape != (7,30,30,3):
            #print(np.array(inputs))
            #print(np.array(inputs).shape)
        self.to_feed = np.hstack([ (1/(255*3))*np.mean( #the 7 inputs
                            np.sum(np.array(inputs),axis=3)#to sum the channels
                                ,axis=(1,2)),#the mean of the square
                                self.speed,self.orient,1]#bias
                        )#.reshape(8,1)#reshaping
        
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
        self.energy -= (0.02*speed+0.2)#not too fast, not still
        #print(round(self.orient*180 / pi %360,0))

    def inc_speed(self,delta):
        new_speed = self.speed + delta
        new_speed = new_speed - (new_speed%max_speed)*(new_speed>=max_speed)
        self.speed = new_speed * (new_speed>=min_speed) + min_speed*(new_speed<min_speed)

    def turn(self,direction=-1):

        self.orient+=direction*(pi/10) 
        self.orient  = (self.orient*180 / pi %360)*pi/180


