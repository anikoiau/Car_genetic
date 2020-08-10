# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:22:25 2020

@author: soumitra
"""


import time
import numpy as np
import pygame
from pygame import Vector2
import math
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from neuralNetwork import forward_propagation, sigmoid, sigmoid_backwards, linear_forward, linear_activate_forward, initialize, predict


np.random.seed()


GEN = 50

width = 1300
height = 750


MS = 3

# im = cv2.imread('Untitled1.png')

# cv2.imshow('ia', im)

# plt.imshow(im)



class Line():
    def __init__(self, start, end):
        self.start = Vector2(start)
        self.end = Vector2(end)
        
        
        

class Ray():
    def __init__(self, pos, direction):
        self.pos = Vector2(pos)
        self.dir = Vector2(direction)
        
        
        
    def intersect(self, line):
        x1 = line.start.x
        y1 = line.start.y
        # end point
        x2 = line.end.x
        y2 = line.end.y
    
        #position of the ray
        x3 = self.pos.x
        y3 = self.pos.y
        x4 = self.pos.x + self.dir.x
        y4 = self.pos.y + self.dir.y
    
        #denominator
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        #numerator
        num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        if den == 0:
            return None
        
        #formulas
        t = num / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    
        if t > 0 and t < 1 and u > 0:
            #Px, Py
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            pot = Vector2(x, y)
            return pot
        
       
        
        
class Car():
    def __init__(self, pos = None, brain = None):
        if pos != None:
            self.pos = Vector2(pos)
            
        else:
            self.pos = Vector2(500, 680)
     
        self.rot = 90
        self.v = Vector2(-MS, 0)
        self.a = Vector2(0, 0)
        self.maxspeed = MS
        self.maxforce = .9
        
        self.score = 0
        self.fitness = 0
        
     
        self.h = 10
        self.w = 5
        self.diag_length = math.sqrt(self.h ** 2 + self.w ** 2)
        
        
        if brain == None:
            
            np.random.seed()
            
            self.brain = initialize([6, 3])
            
        else:
            self.brain = brain
        
        
        
        
        
    def get_points(self):
        
        # self.rot = math.degrees(math.atan2(self.v.y, self.v.x) + math.pi / 2)
        
        self.top = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2) * (self.h/2),\
                                      math.sin(math.radians(-self.rot) + math.pi / 2) * (self.h/2))
        
        self.bottom = self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2) * (self.h/2),\
                                         math.sin(math.radians(-self.rot) + math.pi / 2) * (self.h/2))
            
            
            
        
        self.left = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + math.pi / 2) * (self.w/2),\
                                      math.sin(math.radians(-self.rot) + math.pi / 2 + math.pi / 2) * (self.w/2))
      
        
        self.right =  self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + math.pi / 2) * (self.w/2),\
                                      math.sin(math.radians(-self.rot) + math.pi / 2 + math.pi / 2) * (self.w/2))
        
       
        
        
        
    def diagonal_points(self):
        
        self.topLeft = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 - \
                                                    math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 - \
                                                   math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2)
            
            
            
        self.topLeft2 = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 - \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) - math.pi / 8) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 - \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) - math.pi / 8) * self.diag_length / 2)
            
            
        self.topLeft3 = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 - \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) - math.pi / 7) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 - \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) - math.pi / 7) * self.diag_length / 2)
            
            
            
        self.topRight = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + \
                                                    math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 + \
                                                   math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2)
            
            
            
        self.topRight2 = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) + math.pi / 8) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 + \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) + math.pi / 8) * self.diag_length / 2)
            
            
        self.topRight3 = self.pos - Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) + 2*math.pi / 8) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 + \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) + 2*math.pi / 8) * self.diag_length / 2)
            
        
            
        self.bottomLeft2 = self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) + math.pi / 8) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 + \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) + math.pi / 8) * self.diag_length / 2)
            
            
        self.bottomLeft3 = self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) + 2*math.pi / 8) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 + \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) + 2*math.pi / 8) * self.diag_length / 2)
            
            
            
        self.bottomRight2 = self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 - \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) - math.pi / 8) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 - \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) - math.pi / 8) * self.diag_length / 2)
            
            
            
        self.bottomRight3 = self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 - \
                                                    math.asin((self.w / 2) / (self.diag_length / 2)) - 2* math.pi / 8) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 - \
                                                   math.asin((self.w / 2) / (self.diag_length / 2)) - 2*math.pi / 8) * self.diag_length / 2)
            
            
        
        
            
        self.bottomRight = self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 - \
                                                    math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 - \
                                                   math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2)
            
            
        self.bottomLeft = self.pos + Vector2(math.cos(math.radians(-self.rot) + math.pi / 2 + \
                                                    math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2,\
                                          math.sin(math.radians(-self.rot) + math.pi / 2 + \
                                                   math.asin((self.w / 2) / (self.diag_length / 2))) * self.diag_length / 2)
            
            
            
        self.rectpoints = [self.topLeft, self.topRight, self.bottomRight, self.bottomLeft]
    
    
            
        
        
    def update(self):
        
        if self.v.length() == 0:
            self.v.rotate_ip(180)
                    
        else:
            self.v = self.v + Vector2(-.03 * self.guide)
                  
        
        # if self.v.length() > self.maxspeed:
        #     self.v.scale_to_length(self.maxspeed)
        
        self.pos += self.v
        
        self.a *= 0
        
        self.score += 5 + self.min_dist + (self.v.length() * 2)
        
        
        
        
    def rotate(self, rad):
        
        self.rot += rad
        self.v.rotate_ip(-rad)
        self.guide.rotate_ip(-rad)
        
        
    def draw(self, screen):
        
        pygame.draw.polygon(screen, (255, 0, 0), self.rectpoints)
        
        
       
        
    def rayCretor(self):

        
        self.guide = Vector2(self.top - self.bottom).normalize()
        
        self.rays = []
        
       
        self.rays.append(Ray(self.pos, Vector2(self.guide.rotate(0))))
        self.rays.append(Ray(self.pos, Vector2(self.guide.rotate(45))))
        self.rays.append(Ray(self.pos, Vector2(self.guide.rotate(-45))))
        self.rays.append(Ray(self.pos, Vector2(self.guide.rotate(90))))
        self.rays.append(Ray(self.pos, Vector2(self.guide.rotate(-90))))
            
            
        
            
         
       
        
    def raycasting(self, lines, screen, toggle = False):
        
        distances = []
        
        
        for (key, ray) in enumerate(self.rays):
            
            i = None
            dist = 100000
            for line in lines:
                intersection = ray.intersect(line)
                
                if intersection != None:
                    d = intersection.distance_to(ray.pos)
                    
                    if d < dist:
                        dist = d
                        i = intersection
                        
                        
                
            if i != None:
                
                dis = self.pos.distance_to(i)
                
                distances.append(dis)
                
                
                if toggle == True:
                    pygame.draw.line(screen, (255, 255, 255), (ray.pos.x, ray.pos.y), (i.x, i.y), 1)
                    
                    pygame.draw.circle(screen, (0, 255, 0), (int(i.x), int(i.y)), 2)
                    
                    
            else:
                distances.append(0)
                
                
        self.max_dist = max(distances)
        
        self.min_dist = min(distances)
        
        distances.append(self.v.length())
        
        distances = np.array(distances)
        
        distances = distances / self.max_dist
        
        
        
        return distances, self.min_dist
    
                
               
    
corners = [Vector2(0, 0), Vector2(width - 1, 0), Vector2(width - 1, height - 1), Vector2(0, height -1)]        


borders = [
                Line(corners[0], corners[1]),
                Line(corners[1], corners[2]),
                Line(corners[2], corners[3]),
                Line(corners[3], corners[0])
          ]        



track = []


tr = pd.read_csv('track.csv').values

tr = tr.tolist()

for i in range(1, len(tr) - 1, 2):
    p1 = tr[i]
    p2 = tr[i + 1]
    
    track.append(Line(Vector2(p1), Vector2(p2)))



# trackPoints1 = [
#             (189, 646), (180, 580), (160, 500), (149, 423), (163, 306), (221, 200), (352, 145), (437, 129), (842, 160), (891, 188),
#             (899, 270), (859, 319), (761, 352), (570, 447), (820, 420), (948, 336), (1091, 312), (1205, 362), (1277, 509), (1252, 650), 
#             (1181, 720), (260, 712)
#               ]


# trackPoints2 = [
#             (239, 618), (205, 435), (218, 307), (295, 228), (418, 181), (817, 217), (826, 253), (812, 271), (499, 414), (453, 460),
#             (451, 508), (496, 523), (574, 523), (838, 477), (968, 388), (1080, 376), (1168, 408), (1227, 534), (1208, 601), 
#             (1107, 658), (297, 652) 
#                 ]


# for i in range(1, len(trackPoints1)):
    
#     l = Line(Vector2(trackPoints1[i - 1]), Vector2(trackPoints1[i]))
    
#     track.append(l)
    
    
# track.append(Line(Vector2(trackPoints1[-1]), Vector2(trackPoints1[0])))


# for i in range(1, len(trackPoints2)):
    
#     l = Line(Vector2(trackPoints2[i - 1]), Vector2(trackPoints2[i]))
    
#     track.append(l)
    
    
# track.append(Line(Vector2(trackPoints2[-1]), Vector2(trackPoints2[0])))


def drawtrack(screen):
    for line in track:
        pygame.draw.line(screen, (255, 255, 255), (line.start.x, line.start.y), (line.end.x, line.end.y), 1)
    
    # pygame.draw.lines(screen, (255, 255, 255), True, trackPoints1, 1)
    # pygame.draw.lines(screen, (255, 255, 255), True, trackPoints2, 1)



# top = car.image
screen = pygame.display.set_mode((width, height))


carToggle = True

toggle = False


carSaved = []



def mutate(parameters, rate):

    
    # if np.random.random() < rate:         
    for (key, value) in parameters.items():
        m, n = value.shape
        
        for i in range(m):
            for j in range(n):
                if np.random.random() < rate:
                    r = (np.random.random() * 2 - 1) * rate * .1
                    
                    value[i][j] += r
                 
                    
                
        parameters[key] = value
                    
            
    return parameters




def chooseParent(array, total):
    
    np.random.seed()
    
    r = np.random.randint(0, total)
    
    i = 0
    s = 0
    
    while s < r:
        s += array[i]
        
        
        i += 1
        
   
    return i - 1



    
def callNextGeneration(saved, bestCar, pos):
    
    global GEN
    
    
    total_score = 0
    max_score = 0
    fitness_prob = []
    
    scores = []
    
    for car in saved:
        
        
        total_score += car.score
        
        scores.append(car.score)
        
 
    
    mscore = max(scores)
    if mscore > bestCar.score:
        mindex = scores.index(mscore)
        
    else:
        mindex = -1
    
    
    
    lists = []
    
    for i in range(0, GEN):
        
        if i == 0:
            if mindex == -1:
                br = bestCar.brain
                lists.append(Car(pos = pos, brain = br))
                # print('bestcar', bestCar.score, saved[mindex].score)
                
            else:
                br = saved[mindex].brain
                lists.append(Car(brain = br))
                # print("max score", bestCar.score, saved[mindex].score)
            
        else:    
            np.random.seed()
            
            par1_index = chooseParent(scores, total_score)
            par2_index = chooseParent(scores, total_score)
            
        
            
            
            while par1_index == par2_index:
                par2_index = chooseParent(scores, total_score)
              
                
            
            
            parent1 = saved[par1_index]
            
            parent2 = saved[par2_index]
     
            
            brain1 = parent1.brain
            brain2 = parent2.brain
            
            child_dict = {}
            
            for (key, value) in brain1.items():
                m, n = brain1[key].shape
                child = np.zeros_like(brain1[key])
                
                for i in range(m):
                    for j in range(n):
                        
                        if np.random.random() < .5:
                            child[i][j] = brain1[key][i][j]
                            # print('1')
                        else:
                            child[i][j] = brain2[key][i][j]
                            # print('2')
                
                
                            
                child_dict[key] = child
           
            child_dict = mutate(child_dict, .3)
        
            lists.append(Car(pos = pos, brain = child_dict))   
    
    
    
    return lists
    


# X = np.zeros((1, 6))

# X = np.genfromtxt('1st.csv', delimiter=',')


# W1 = np.genfromtxt('w1.csv', delimiter=',')
# W2 = np.genfromtxt('w2.csv', delimiter=',')
# b1 = np.genfromtxt('b1.csv', delimiter=',')
# b2 = np.genfromtxt('b2.csv', delimiter=',')

# parameters = {'W1' : W1, 'W2' : W2, "b1" : b1, 'b2' : b2}

# b1 = b1.reshape((15, 1))
# b2 = b2.reshape((3, 1))

W1 = np.genfromtxt('W1_genetic.csv', delimiter=',')
b1 = np.genfromtxt('b1_genetic.csv', delimiter=',')


b1 = b1.reshape((3, 1))

param = {'W1' : W1, 'b1' : b1}



cars = []

for i in range(GEN):
    
    np.random.seed(int(time.time()))
    
    cars.append(Car())
    
    
    

c = 0

count = 0


carToggle = False


points = []



turn = 8

generations = 0


currentMax = 0


frames = 1


start_time = int(time.time())

while True:
    
    t = False
    
    pygame.time.Clock().tick(100)
    
    frames += 1
    
    drawtrack(screen)
    
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            
    
        if e.type == pygame.MOUSEMOTION or e.type == pygame.MOUSEBUTTONUP:
            if count % 2 == 1:
                end_pos = Vector2(e.pos)
                pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 1)
                
                pygame.display.update()
                
                # print('inside mousemotion', count, start_pos, end_pos)
            
        if e.type == pygame.MOUSEBUTTONDOWN:
            
            if e.button == 3:
                count += 1
        
            if e.button == 1:
                if count % 2 == 0:
                    start_pos = Vector2(e.pos)                   
                    
                    count += 1
                    
                else:
                    end_pos = Vector2(pygame.mouse.get_pos())
                    pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 3) 
                    count += 1
                    
                    track.append(Line(start_pos, end_pos))
                      
    
    screen.fill((0, 0, 0))
    
    keys = pygame.key.get_pressed()
    
    
    
    drawtrack(screen)
      
    if keys[pygame.K_SPACE]:
        p = pygame.mouse.get_pos()
        
        carToggle = True
        
        pos = Vector2(p)
        
        for car in cars:
            car.pos = Vector2(pos)
        
    if keys[pygame.K_a]:
        t = True
    

        
    if carToggle == True:  
        
        c += 1
       
        # print((int(time.time()) - start_time))
        
        # end_time = int(time.time())
        
        # if (end_time - start_time) % 100 == 0:
            
        #     carSaved.extend(cars)
        #     bestcar = car
        #     del cars[:]
        #     cars = callNextGeneration(carSaved, bestcar, pos)
            
        #     generations += 1
            
        #     print(generations, bestcar.score, 'got impatient')
            
            
            # continue
            
        
        # print(frames)
        
        if frames > 3000:
            frames = 0
            carSaved.extend(cars)
            bestcar = car
            del cars[:]
            cars = callNextGeneration(carSaved, bestcar, pos)
            
            generations += 1
            
            print(generations, bestcar.score, 'got impatient')
            
            continue
        
        
        for car in cars:
            car.get_points()
            car.diagonal_points()
            car.rayCretor()
        
            
            if currentMax < car.score:
                currentMax = car.score
                bestcar = Car(brain = car.brain)
                bestcar.score = car.score
                # print('hello')
        
            # if keys[pygame.K_LEFT]:
            #     car.rotate(turn)

            # elif keys[pygame.K_RIGHT]:
            #     car.rotate(-turn)
      
            
            # if keys[pygame.K_RIGHT]:
            #     turn += .5
                
                
            # elif keys[pygame.K_LEFT]:
            #     turn -= .5
                
            # if keys[pygame.K_UP]:
                
            #     car.v = car.v + Vector2(.05 * car.guide)
                
            
            # elif keys[pygame.K_DOWN]:
            #     if car.v.length() == 0:
            #         car.v.rotate_ip(180)
                    
            #     else:
            #         car.v = car.v + Vector2(-.05 * car.guide)
        
            
            inputs, min_dist = car.raycasting(track, screen, t)
            
            
            inputs = inputs.reshape((6, 1))
            
            
            activations = predict(inputs, car.brain)
            
            
            # print(activations)
            
            if activations == 0:
                car.rotate(turn)
                
            elif activations == 1:
                    car.rotate(-turn)
                    
            else:
                car.v = car.v + Vector2(.05 * car.guide)

            
            if min_dist < (car.v.length() + .5)  or car.v.length() < .8:
                carSaved.append(car)
                cars.remove(car)
                # print(min_dist, generations)
                continue
         
            
            if car.score > 150000 * 1.5:
                carSaved.extend(cars)
                bestcar = car
                del cars[:]
                cars = callNextGeneration(carSaved, bestcar, pos)
                
                generations += 1
                
                frames = 0
                
                print(generations, bestcar.score, 'max reached')
                
            if keys[pygame.K_UP]:
                car.maxspeed += .05      
                
            
            car.update()
        
            car.draw(screen)
            
            # print(car.score)
          
            
        if len(cars) == 0:
            cars = callNextGeneration(carSaved, bestcar, pos)
            del carSaved[:]
            
            generations += 1
            
            frames = 0
                
            print(generations, bestcar.score, 'cars dead')


            

    pygame.display.update()
    

params = bestcar.brain

W1_genetic = params['W1']
b1_genetic = params['b1']

np.savetxt('W1_genetic.csv', W1_genetic, delimiter=',')
np.savetxt('b1_genetic.csv', b1_genetic, delimiter=',')
# np.savetxt('1st.csv', X, delimiter = ',')


# sum(X[:, -1] == 0)
# sum(X[:, -1] == 1)
# sum(X[:, -1] == 2)



# l = []
# r = []
# f = []

# for i in range(X.shape[0]):
#     if X[i, -1] == 0:
#         l.append(X[i])
        
#     elif X[i, -1] == 1:
#         r.append(X[i])
        
#     else:
#         f.append(X[i])

# f = f[:3060]
# l = l[:3060]


# X1 = np.vstack((l, r, f))

# np.random.shuffle(X1)


# np.savetxt('1.csv', X1, delimiter = ',')

# # indices = np.argwhere(X[:, -1] == 2)



# # max(np.argwhere(X[:, -1] == 0)[:, 0])
# # max(np.argwhere(X[:, -1] == 1)[:, 0])
# # max(np.argwhere(X[:, -1] == 2)[:, 0])


# # a = Vector2(0, -1)

# # math.degrees(math.atan2(a.y, a.x) + math.pi/2)



single = []

for line in track:
    single.append((line.start.x, line.start.y))
    single.append((line.end.x, line.end.y))
    


a = np.array(single)

np.savetxt('track.csv', a, delimiter=',')


