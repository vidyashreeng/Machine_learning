#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
cluster1 = []
cluster2 = []
cluster3 = []
x1,y1 = 6.2,3.2
x2,y2 = 6.6,3.7
x3,y3 = 6.5,3.0
points = [[5.9,3.2],
           [4.6,2.9],
           [6.2,2.8],
           [4.7,3.2],
           [5.5,4.2],
           [5.0,3.0],
           [4.9,3.1],
           [6.7,3.1],
           [5.1,3.8],
           [6.0,3.0]]

clist = []
def eucledian_distance_1(a,b,xc1,yc1,xc2,yc2,xc3,yc3):
   

    dis1 = math.sqrt((a-xc1)**2 + (b-yc1)**2)
    dis2 = math.sqrt((a-xc2)**2 + (b-yc2)**2)
    dis3 = math.sqrt((a-xc3)**2 + (b-yc3)**2)
    mind = min(dis1,dis2,dis3)
    if(mind == dis1):
        clist.append(1)
        cluster1.append([a,b])
    elif(mind==dis2):
        clist.append(2)
        cluster2.append([a,b])
    else:
        clist.append(3)
        cluster3.append([a,b])
   
       
def fnc(cls):
    sumi = 0
    sumj = 0
    for i,j in cls:
        sumi = sumi + i
        sumj = sumj + j
    ncx = (sumi/len(cls))
    ncy = (sumj/len(cls))
    return ncx,ncy

for i,j in points:
    eucledian_distance_1(i,j,x1,y1,x2,y2,x3,y3)
   
print(clist)

clist = []
nxc1,nyc1 = fnc(cluster1)
nxc2,nyc2 = fnc(cluster2)
nxc3,nyc3 = fnc(cluster3)

print("\nNew centroids are")
print("cluster1",nxc1,nyc1)
print("cluster2",nxc2,nyc2)
print("cluster3",nxc3,nyc3)

cluster1 = []
cluster2 = []
cluster3 = []

for i,j in points:
    eucledian_distance_1(i,j,nxc1,nyc1,nxc2,nyc2,nxc3,nyc3)
   

print("\n",clist)

clist = []

nxc1,nyc1 = fnc(cluster1)
nxc2,nyc2 = fnc(cluster2)
nxc3,nyc3 = fnc(cluster3)

print("\nNew centroids are")
print("cluster1",nxc1,nyc1)
print("cluster2",nxc2,nyc2)
print("cluster3",nxc3,nyc3)

cluster1 = []
cluster2 = []
cluster3 = []

for i,j in points:
    eucledian_distance_1(i,j,nxc1,nyc1,nxc2,nyc2,nxc3,nyc3)
print("\n",clist)
clist = []

nxc1,nyc1 = fnc(cluster1)
nxc2,nyc2 = fnc(cluster2)
nxc3,nyc3 = fnc(cluster3)

print("\nNew centroids are")
print("cluster1",nxc1,nyc1)
print("cluster2",nxc2,nyc2)
print("cluster3",nxc3,nyc3)

cluster1 = []
cluster2 = []
cluster3 = []

for i,j in points:
    eucledian_distance_1(i,j,nxc1,nyc1,nxc2,nyc2,nxc3,nyc3)


# In[ ]:




