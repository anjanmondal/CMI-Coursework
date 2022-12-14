# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:06:42 2022

@author: anjan
"""

books={}
borrowers={}
checkouts={}

#curr=input()
i=0
while i<3:
    
    curr=input()
    while True:
        curr=input()
        if curr=='Borrowers':
            break      
        curr=curr.split('~')
        books[curr[0]]=curr[1]  # Accession Number~Title
    i+=1
    # curr=input()
    while True:
        curr=input()
        if curr=='Checkouts':
            break
        
        curr=curr.split('~')
        borrowers[curr[0]]=curr[1]   #Username~FullName
    i+=1
    # curr=input()
    while True:
        curr=input()
        if curr=='EndOfInput':
            break
        curr=curr.split('~')
        checkouts[(curr[0],curr[1])]=curr[2] #Username~AccessionNumber~DueDate
    i+=1

## Output ->Due Date~Full Name~Accession Number~Title

usernames=list(checkouts.keys())

outlist=[]

for k in usernames:
    due_date=checkouts[k]
    full_name=borrowers[k[0]]
    acc_num=k[1]
    title=books[acc_num]   
    row=due_date,full_name,acc_num,title
    outlist.append(row)

outlist.sort()

outlist_final=[]
for row in outlist:
    outlist_final.append(row[0]+'~'+row[1]+'~'+row[2]+'~'+row[3])
for row in outlist_final:
    print(row)
