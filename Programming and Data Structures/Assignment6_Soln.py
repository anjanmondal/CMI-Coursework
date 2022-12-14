# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:26:22 2022

@author: anjan
"""

N=int(input())

r1=input()
r2=input()

r1=r1.split()
r2=r2.split()
r1=tuple([int(i) for i in r1])
r2=tuple([int(i) for i in r2])


memo={}

def domino(r1,r2):
    
    if (r1,r2) in memo.keys():
        return memo[r1,r2]
    #print(r1)
    N=len(r1)
    if N==1:
        if (r1,r2) not in memo.keys():
            ans=abs(r1[0]-r2[0])
            memo[r1,r2]=ans
            return memo[r1,r2]
    elif N==2:
        if (r1,r2) not in memo.keys():
            ans=max(domino(r1[:1],r2[:1])+domino(r1[1:],r2[1:]),abs(r1[0]-r1[1])+
                    abs(r2[0]-r2[1]))
            memo[r1,r2]=ans
            return memo[r1,r2]
    elif (r1,r2) not in memo.keys():
        ans=0
        for i in range(2,N):
                
            if (r1[:i+1],r2[:i+1]) not in memo.keys():
                memo[r1[:i+1],r2[:i+1]]=max(domino(r1[:i],r2[:i])
                + domino(r1[i:i+1],r2[i:i+1]), domino(r1[:i-1],r2[:i-1])
                + domino(r1[i-1:i+1],r2[i-1:i+1]))

        return memo[r1,r2] 



print(domino(r1,r2))
