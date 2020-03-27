# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:38:31 2020

radar grid map for zhihu 

@author: wenzt
"""
import numpy as np
import math
import matplotlib.pyplot as plt

#parameters
##environment
worldsize = [50, 50]#rectangle
gridsize = 0.08 #resolution of grid map

#sensor
MAX_dect_range = 20
alpha = 2.5 # angle resolution
beta = 0.2 # range resolution 

#experiment data
pose = np.matrix([[18.12, 23.42,  1.937]]) #x(m),y(m),orientation(rad) pose in Vicon coordinate system
measurement = np.load("scan.npy") # each row [x, y, range, angle] range, angle are measured by radar, (x,y) is calculated  

### functions of mapping
def updatemap_beam(gridsize,p_plate,scanpoint,grid,alpha,beta,MAX_dect_range):
    
    origin_sx = int(p_plate[0,0]/gridsize)
    origin_sy = int(p_plate[0,1]/gridsize)
        
    for nump in range(len(scanpoint[:,0])):
        occ_x = scanpoint[ nump , 0]
        occ_y = scanpoint[ nump , 1]
        searchx_min = min(origin_sx,int(occ_x/gridsize))
        searchx_max = max(origin_sx,int(occ_x/gridsize))
        searchy_min = min(origin_sy,int(occ_y/gridsize))
        searchy_max = max(origin_sy,int(occ_y/gridsize))
        for jx in range(max(searchx_min-5,0),min(searchx_max+5,len(grid[:,0]))):
            for jy in range(max(searchy_min-5,0),min(searchy_max+5,len(grid[:,0]))):
                cellx = jx*gridsize + gridsize/2  
                celly = jy*gridsize + gridsize/2
                
                scan_single_point = scanpoint[nump,:]
                phi = math.atan2(celly-p_plate[0,1],cellx-p_plate[0,0])
                
                #calculate difference between two angles
                lphi = [ math.cos(phi) , math.sin(phi) ] #计算角度向量
                theta = scan_single_point[0,3] + p_plate[0,2]
                ltheta = [ math.cos(theta) , math.sin(theta) ]
            
                if lphi[0] * ltheta[0] + lphi[1] * ltheta[1] > 1:
                    dangle = math.acos( lphi[0] * ltheta[0] + lphi[1] * ltheta[1]  - 1e-5)
                elif lphi[0] * ltheta[0] + lphi[1] * ltheta[1] < -1:
                    dangle = math.acos( lphi[0] * ltheta[0] + lphi[1] * ltheta[1]  + 1e-5)
                else:
                    dangle = math.acos( lphi[0] * ltheta[0] + lphi[1] * ltheta[1] )
                
                if abs(dangle) < alpha/2/180*math.pi:            
                    flag,l = inverse_sensor_model(scanpoint[nump,:],MAX_dect_range,cellx,celly,p_plate,beta,gridsize,dangle)
                    grid[jx,jy] = grid[jx,jy] + l    
    
    return grid

def inverse_sensor_model(scan_single_point,MAX_detect_range,cellx,celly,p_plate,beta,gridsize,dangle):
    
    flag = 2
    r = scan_single_point[0,2]
    
    deltar = (2)**0.5 * gridsize
    deltat = (2)**0.5 * gridsize / r
    pd = 0.8
    sigmar = 0.07/3
    sigmat = 0.5/180*3.14
    rgrid = math.sqrt((cellx-p_plate[0,0])**2 + (celly-p_plate[0,1])**2)
    pr = 0.5 * ( math.erf( (rgrid + deltar - r)/(1.414 * sigmar) ) - math.erf( (rgrid - deltar - r)/(1.414 * sigmar) ) )
    pt = 0.5 * ( math.erf( (dangle+ deltat)/(1.414 * sigmat) ) - math.erf( ( dangle- deltat)/(1.414 * sigmat) ) )
    focc = pr * pt
    femp = math.exp(-rgrid**2/(2*(r/4)**2)) * pt
    p = 0.5*( 1 + pd * (focc - femp) )
     
    if rgrid >= MAX_detect_range:
        flag = 2
    else:
        if abs(rgrid-r) < beta/2 and r < MAX_detect_range:
            flag = 1
        elif rgrid < scan_single_point[0,2]:
            flag = 0
        else:
            flag = 2                    
    
    if flag == 1 or flag == 0:
        l = math.log10(p/(1-p))
    else:
        l = 0
    
    return flag,l#flag标记该cell占用情况，1-occupied，0-free，2-undetected

#### main
g0 = np.matrix(np.zeros((int(worldsize[0]/gridsize),int(worldsize[1]/gridsize)))) #initialize
g0 = updatemap_beam(gridsize,pose,np.matrix(measurement),g0,alpha,beta,20)#update the grid map using measurement

#invert log-odds into prob.
for jx in range(int(worldsize[0]/gridsize)):
    for jy in range(int(worldsize[1]/gridsize)):
        g0[jx,jy] = 1-1/(1+10**(g0[jx,jy]))

#figure of map
plt.figure(figsize = (10,10))
plt.imshow(g0.T,origin='lower')
plt.colorbar()
tx = np.linspace(0,len(g0[:,0]),5)
ty = np.linspace(0,len(g0.T[:,0]),5)
rx = np.linspace(0,worldsize[0],5)
ry = np.linspace(0,worldsize[1],5)
rx = [round(i,2) for i in rx]
ry = [round(i,2) for i in ry]
plt.xticks(tx,rx)
plt.yticks(ty,ry)