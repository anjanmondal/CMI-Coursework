# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:13:33 2022

@author: anjan
"""
# Write a function contracting(l) that takes as input a list of integer l and returns True if
#  the absolute difference between each adjacent pair of elements strictly decreases
def contracting(l):
    diff=abs(l[1]-l[0])
    i,j=1,2
    while j<len(l):
        if abs(l[j]-l[i])<diff:
            diff=abs(l[j]-l[i])
            i+=1
            j+=1
        else:
            return False
    return True

def matrixflip(mat,direction):
    mat_c=[row[:] for row in mat]
    if direction=="h":
        for i in range(len(mat_c)):
            for j in range(len(mat_c[0])):
                mat_c[i][j]=mat[i][-(1+j)]
        
    if direction=="v":
        mat_c.reverse()
        
    return mat_c
            

def orangecap(matches):
    total_scores={}
    for match in matches:
        for player in matches[match]:
            total_scores[player]=total_scores.get(player,0)+matches[match][player]
    
    keymax = max(total_scores, key= lambda x: total_scores[x])
    
    return keymax,total_scores[keymax]


def append_list(item):
    app=[]
    if type(item)!="list":
        app.append(item)
    else:
        append_list(item)
        
    return app
        
    
def flatten(item):

    fin_l=[]
    if type(item)==list:
        for i in item:
            for x in flatten(i):
                fin_l.append(x)
                #print(x)
    elif type(item)!=list:
        fin_l.append(item)
        #print(item)
    return fin_l
