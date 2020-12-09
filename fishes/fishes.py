import numpy as np
import cv2
import random as rand
import math
from math import pi
import matplotlib.pyplot as plt
from fish import Fish
from alivefood import AliveFood
from utils import merge_images_h,command_bridge,merge_images_v,get_subwindow

def dclick(event, x, y, flags, param):
        global refPt, focus,mpos
        mpos = np.array([x,y])
        if event == cv2.EVENT_LBUTTONDBLCLK:
            
            refPt = np.array([x, y])
            focus=True
            print('clicked',focus,refPt)
refPt=None
mpos=None
def run_simulation(fishes=None,
                    foods=None,
                    n_pop=10,
                    max_time=3000,
                    verbose=True,
                    size_w = (600,900,3),
                    initial_simulation=False,
                    epoch=None,
                    record=False,
                    ALIVE=False,
                    folder = 'frames',
                    num_food=None):
    global refPt
    #WORLD definition
    size_w = size_w
    size_w1,size_w2 = size_w[:2][::-1]

    #the world that the food percieve, it contains only the fishes
    food_world = (np.zeros(size_w)+(255,255,255)).astype(np.uint8)
    #the world that the fishes percieve, it contains only the food
    fish_world = 255*np.ones(size_w,dtype=np.uint8)
    if initial_simulation and verbose:
        #tutorial
        cm_br = command_bridge(f_size=0.7,commands=['d debug',
                                            'e energy',
                                            'click to focus on one',
                                            'b to focus the best'
                                            'esc to remove focus',
                                            'h increse speed',
                                            'k decrese speed',
                                            'esc again to exit'])
        cv2.imshow('beginning',merge_images_h([cm_br]))
        res = cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Mouse callback
    #refPt = (10,10)
    
            

    if verbose and not record:
        cv2.namedWindow("Simulation")
        cv2.setMouseCallback("Simulation", dclick)


    time = 0
    max_radius = 40
    n_pop = n_pop#30

    s1= 100
    s2 = 100
    fishes_pos_orient_color = [ (  rand.randint(s2,size_w2-s2),#x
                            rand.randint(s1,size_w1-s1),#y
                            2*pi*rand.random(), #orientation 
                            np.random.choice(range(256), size=3, replace=False)#,dtype=np.uint8)#color
                        ) for i in range(n_pop)]
    if fishes is None:
        fishes = [Fish(orient=o,pos=(p1,p2),color=c) for p1,p2,o,c in fishes_pos_orient_color]
    else:
        fishes = fishes
    #fishes = [Fish()]
            #,Fish(pos=(100,102),orient=pi*1)]
    if foods is None:
        if num_food is None:
            num_food = int(n_pop*3/2)+1
        print(size_w1,size_w2)

        foods_pos_orient=np.array([(  rand.randint(s2+1+max_radius,size_w2-s2-1-max_radius),#0-900
                                rand.randint(s1+1+max_radius,size_w1-s1-1-max_radius),#0-600
                                2*pi*rand.random())
                                         for i in range(num_food)])
        foods = [AliveFood(pos=(p1,p2),orient=o) for p1,p2,o in foods_pos_orient]
    else:
        foods = foods
    dead_foods = []
    food_positions = np.array([f.pos for f in foods ])#.astype(np.uint8)
    #
    num_foods = []
    alive = n_pop
    debug = False
    energy = False

    selected = None
    focus = False
    focus_best = focus
    food_focus = True
    food_selected = None
    while(time<max_time):
    
        #get focus on individual
        #subject to verbose
        if verbose:
            if refPt is not None:
                focus_best = False
                focus = True
                print('clicked on ',refPt[::-1])
                all_dist = np.sqrt(np.sum((np.array([f.pos+1000*(f.energy<=0) for f in fishes])-(refPt[::-1]+np.array([s1,s2])))**2,axis=1))
                selected = fishes[np.argmin(all_dist)]
                refPt=None

        

            
        #food steps
        #create food
        #add food every 3 steps
        if (not ALIVE or False)  and time % 20 ==1: #and food_positions.shape[0]<3*alive: 
            pos1,pos2,o = (rand.randint(s2+1+max_radius,size_w2-s2-1-max_radius),#0-900
                            rand.randint(s1+1+max_radius,size_w1-s1-1-max_radius),#0-600
                     2*pi*rand.random())
            mind=int(rand.random())
            if mind==0 and len(dead_foods)>0:
                pool=dead_foods
            else:
                pool=foods
            new_food = AliveFood(pos=(pos1,pos2),orient=o,mind=pool[rand.randint(0,len(pool)-1)].mind)
            foods.append(new_food)
            #insert in array to calculate distancies
            food_positions = np.vstack([food_positions,[pos1,pos2]])
            
            

        num_foods.append(len(food_positions))

        #draw food
        fish_world = (np.zeros(size_w)+(255,255,255)).astype(np.uint8)
        for i,food in enumerate(foods):
            if ALIVE:
                food.get_perception(food_world)
                react = food.predict(food.to_feed)
                #if food==food_selected:
                    #print(food.to_feed,react)
                food.turn(direction=react[0])
                food.inc_speed(delta=react[1])
                food.move(fish_world,dx=s1-food.s1,dy=s2-food.s2)
                food_positions[i]=food.pos
                food.outout = react
            food.draw(fish_world)
            
        food_world =  (np.zeros(size_w)+(255,255,255)).astype(np.uint8)
        #reset food_world after that the fishes have been drawn s.t. the food can understand where they are
        #
        #get perceptions, calculate output
        # increase energy if eat somehtin
        for f in fishes:
            if f.energy>0:
                #f.move(food_world,dx=s1-f.s1,dy=s2-f.s2)
                #getting perceptions
                f.get_perception(fish_world)
                #print(f.to_feed)
                react = f.predict(f.to_feed)
                f.outout = react
                f.turn(direction=react[0])
                f.inc_speed(delta=react[1])   
                f.diversity_speed[int(np.sign(np.round(react[1],1)))+1]+=1
                f.diversity_turn[int(np.sign(np.round(react[0],1)))+1]+=1

                

                if food_positions.shape[0]>0:
                    all_dist = np.sqrt(np.sum((food_positions-f.pos)**2,axis=1))
                    closest_idx = np.argmin(all_dist)
                    min_dist = all_dist[closest_idx]
                if (min_dist<6):
                    #print(min_dist)
                    f.energy +=50
                    f.eaten +=1
                    dead_foods.append(foods.pop(closest_idx))
                    if len(foods) == 0:
                        break
                    food_positions = np.delete(food_positions,closest_idx,axis=0)
                    
                    
        #early stopping for bad individuals
        alive = len([f for f in fishes if f.energy>0])
        if alive<=0 or len(foods)==0:
            break

        
        #draw individuals            
        for f in fishes:
            if f.energy>0:
                f.draw(food_world)
        #res = merge_images_h([fish_world,food_world])
        
        
        
            #subject to verbose
            if verbose:        
                    #print closest food ifdebug activated
                    if debug and food_positions.shape[0]>0:
                        all_dist = np.sqrt(np.sum((food_positions-f.pos)**2,axis=1))
                        #for fp,d in zip(food_positions,all_dist):
                        #    print(fp,d,f.pos[::-1])
                        closest_idx = np.argmin(all_dist)
                        min_dist = all_dist[closest_idx]
                        
                        fp = (int(food_positions[closest_idx][0]),int(food_positions[closest_idx][1]))
                        
                        cv2.circle(food_world,fp[::-1],1,(0,0,255),5)
                        cv2.line(food_world,tuple(f.pos[::-1]),fp[::-1],(100,30,100),1)

        res = np.minimum(fish_world,food_world) 
               
        #print energy panel
        if energy:
            f_sorted = sorted(fishes, key=lambda f: f.energy, reverse=True)
            energies, colors = zip(*[('energy '+str(round(f.energy,2)),f.color) for f in f_sorted ])
            cb = command_bridge(f_size=0.7,commands=energies, colors=colors)
            res = merge_images_h([res,cb])  

        #print focused
        if verbose and food_focus:
            #find best food
            idx_best = np.argmax([f.lifetime for f in foods])
            #select best food
            food_selected = foods[idx_best]
            #print squares for perception on the food_world+fishworld
            food_selected.print_perception(res)
            #get the focused area to show it
            focus_area = get_subwindow(res,pt=food_selected.pos[::-1],deltax=75,deltay=75)
            #draw a rectangle on it
            #cv2.rectangle(res,(int(food_selected.pos[1])-food_selected.s1,
            #                    int(food_selected.pos[0])-food_selected.s2),
            #                         (int(food_selected.pos[1])+food_selected.s1,
            #                    int(food_selected.pos[0])+food_selected.s2),
            #                         (244,0,0))
            # plancia for the food
            cm_br = command_bridge(f_size=0.4,commands=['Life Time '+str(food_selected.lifetime),'Pos '+str(food_selected.pos),
                                                        'in'+str([round(e,1) for e in food_selected.to_feed]),
                                                        'Out: turn '+str(np.round(food_selected.outout[0],2))+
                                                        'Out: speed '+str(np.round(food_selected.outout[1],2))])
            # mergeimages to show
            detail = merge_images_v(food_selected.percs+[focus_area,cm_br])
            res = merge_images_h([res,detail])
        if focus:
            if focus_best:
                idx_best = np.argmax([f.energy for f in fishes])
                selected = fishes[idx_best]
            if selected.energy>0:
                focus_area = get_subwindow(res,selected.pos[::-1],deltax=75,deltay=75)
                
                cm_br = command_bridge(f_size=0.4,commands=['S '+str(selected.speed),
                                            'E '+str(round(selected.energy,1)),
                                            #'O '+str(round(selected.orient*180 / pi %360,0))
                                            ]+
                                            ['in'+str([round(e,1) for e in selected.to_feed])]+
                                            ['Out: turn '+str(np.round(selected.outout[0],2))+
                                            ' Diversity '+str(selected.diversity_turn),
                                            'Out: speed '+str(np.round(selected.outout[1],2))+
                                            ' Diversity '+str(selected.diversity_speed),
                                            ]
                                            
                                            )
                selected.print_perception(res)
                detail = merge_images_v(selected.percs+[focus_area,cm_br])
                res = merge_images_h([res,detail])
            else:
                selected = None
                focus=False
                focus_best = focus
                
        for f in fishes:
            if f.energy>0:
                f.move(food_world,dx=s1-f.s1,dy=s2-f.s2)
        #subject to verbose
        if verbose and not record:
            k = cv2.waitKey(33)# & 0xff
            print('keypressed',k)
            #81 left
            #82 up
            #84 down
            #83 right
            #32 space
            #100 d
            #101 e
            #27 esc
            
            
            if focus and selected!=None:
                if k in [104,107]:
                    direct = -1 if k==104 else 1
                else:
                    direct=0
                selected.turn(direction = direct)
            #for f in fishes:
            #    if f!=selected:
            #        f.turn(direction = rand.random()*2-1)
            if k==98:
                if not focus_best:
                    focus_best=True
                    if not focus:
                        focus=True
                else:
                    focus=False
                    focus_best=False
                #idx_best = np.argmax([f.energy for f in fishes])
                #selected = fishes[idx_best]
            if k==100:
                debug=not debug
            if k==101:
                energy = not energy
            if k==32:
                cv2.waitKey(0)
            #print('pressed',k)
            if k in [106,108]:
                delta = -1 * (k-107)
                print('increasing')
                if focus:
                    selected.inc_speed(delta)
                else:
                    for f in fishes:

                        f.inc_speed(delta)
            
            if k==27:
                if focus:
                    focus=not focus
                    focus_best = False
                else:
                    break

        if verbose==3 :
                res = merge_images_v([
                    merge_images_h([command_bridge(commands=(['Generation '+str(epoch)])),
                                    command_bridge(commands=['Time left '+str(max_time-time)]),
                                    command_bridge(commands=['Avg speed '+str(np.mean([f.speed for f in fishes if f.energy>0]))]),
                                    command_bridge(commands=(['Alive '+str(alive)]))
                                    
                                ]),
                    
                    res])
        if record:
            name = 'Generation_%03d' % epoch + '_time_%03d' %time + '.jpg'
            cv2.imwrite(folder+'/'+name,res)
        elif verbose :
             cv2.imshow('Simulation',res)
        '''if time ==0:
            paths = 255*np.zeros(food_world.shape)
        if time<300:
            for i in range(3):
                paths[:,:,i] += food_world[:,:,i]# np.minimum(food_world,food_world)
            cv2.imshow('path',merge_images_h([paths]))
            print(paths.shape)'''
            
            

        time+=1

        '''if time %300 == 299:
            name = 'paths.jpg'
            
            paths = np.zeros(food_world.shape)
            break'''
    cv2.destroyAllWindows()

    if ALIVE:
        return num_foods,fishes,dead_foods+foods
    else:
        return num_foods,fishes
