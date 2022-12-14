# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 18:15:35 2022
@author: anjan
"""

class Node:
    def __init__(self):
        self.value=None
        self.next=None
        

class Deque:
    def __init__(self):
        newnode=Node()
        self.head=newnode
        self.tail=newnode
        
    def isempty(self):
        return self.head.value==None
    
    def insertfront(self,v):
        if self.isempty():
            self.head.value=v
        else:
            newnode=Node()
            newnode.value=self.head.value
            newnode.next=self.head.next
            self.head.value=v
            self.head.next=newnode
            self.tail=newnode
            while self.tail.next != None:
                self.tail = self.tail.next

            
    def __str__(self):
        if self.head.value == None:
          return(str([]))
        else:
          ptr = self.head
          myl = [ptr.value]
          while ptr.next != None:
            ptr = ptr.next
            myl = myl + [ptr.value]
          return(str(myl))
      
    def insertrear(self,v):
        if self.isempty():
            self.tail.value=v
        else:
            newnode=self.tail
            self.tail=Node()
            newnode.next=self.tail
            self.tail.value=v
            
    def deletefront(self):
        if not self.isempty():
            v=self.head.value
            if self.head.next is not None:
                self.head=self.head.next
            else:
                newnode=Node()
                self.head=newnode
                self.tail=newnode
            return v
    def deleterear(self):
        if not self.isempty():
            v=self.tail
            self.tail=self.head
            if self.head.next is not None:
                while self.tail.next!=v:
                    self.tail=self.tail.next
                self.tail.next=None
            else:
                newnode=Node()
                self.head=newnode
                self.tail=newnode
                
            return v.value
