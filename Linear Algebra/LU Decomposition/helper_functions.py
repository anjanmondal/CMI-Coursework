# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:15:08 2023

@author: anjan
"""
import numpy as np

def swap_rows(A,p,q):
    """
    Parameters
    ----------
    A: A numpy.ndarry matrix of any dimensions
    
    p,q : integers
        The indices of two rows of the matrix.

    Returns
    -------
    numpy.ndarry matrix A with row p swapped with row q.

    """
    
    A[[p,q],:]=A[[q,p],:]
    
    return A

def partial_pivot(L,P,U,k,max_index):
    """
    
    Parameters
    ----------
    max_index: integer
        The index of the maximum of the absolutes
        in the elements below the pivot
    L : square matrix
        A Lower Triangular Matrix
        in the intermediate stage of a Gaussian Elimination
    P : square matrix
        A permuation matrix
    U : square matrix
        The matrix in the intermediate stage of Gaussian Elimination
        that would eventually return an Upper triangular matrix
    k : integer
        The row index of the pivot of the matrix U

    Returns
    -------
    numpy.ndarry matrices L,P,U with rows pivoted

    """
    U[:,k:]=swap_rows(U[:,k:],k,max_index)
    P=swap_rows(P,k,max_index)
    L[:,:k]=swap_rows(L[:,:k],k,max_index)
    
    return L,P,U

def GE_step(L,U,k,j):
    # This function performs the Gaussian
    # Elimination step for row j and column k
    
    L[j,k]=U[j,k]/U[k,k]
    U[j, k: ]=U[j,k:]-L[j,k]*U  [k,k:]
    return L,U

def LU_factorize(A):
    """
    

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.

    Returns
    -------
    L,U,P : 2D Numpy arrays which satisfy LU=PA and are obtained
            using Gaussian Elimination with partial pivoting
     

    """
    n=A.shape[0]
    A=A.astype(float)
    L=np.identity(n)
    P=np.identity(n)
    U=A.copy()
    pivots=[]
    
    for k in range(n):
        max_index = k + np.argmax(np.abs(U[k:, k]))
       
        if U[max_index,k]!=0:
            pivots.append(U[max_index,k])
            
        else:
            return ("The matrix is singular and the rank is at least {}"
                    .format(len(pivots)))
   
        if max_index != k:
              L,P,U=partial_pivot(L,P,U,k,max_index)

        for j in range(k+1,n):
           L,U=GE_step(L,U,k,j)
    return L,U,P


def substitution(L,U,P,b):
    """
    

    Parameters
    ----------
    L : 2D Numpy Array
        A lower triangular Matrix.
    U : 2D Numpy Array
        .
    P : 2D Numpy Array
        A permutation Matrix
    b : 1D Numpy Array
        A vector which satisfies the relation LUx=Pb

    Returns
    -------
    x_ : 1D Numpy Array
        Returns the vector x_ for which LUx_=Pb is true

    """
    n=len(b)
    # Solving PLU=PB
    b_=np.matmul(P,b)
    #Solve Ly=Pb_ using forward substitution
    # Get y=y_
    y_=np.zeros(n)
    
    for i in range(len(y_)):
        if i==0:
            y_[i]=b_[0]
        else:
           y_[i]=(b_[i]-np.matmul(L[i,:i],y_[:i]))/(L[i,i])
    
    
    #Solve Ux=y_ using backward substitution
    # Get x=x_
    x_=np.zeros(n)
    for j in range(n-1,-1,-1):
        if j==n-1:
            x_[j]=y_[j]/U[j,j]
        else:
            x_[j]=(y_[j]-np.matmul(U[j,j+1:],x_[j+1:]))/U[j,j]
    
    return x_

def p_piv(p):
    """
    

    Parameters
    ----------
    p : 2D Numpy array
        Permutation Matrix

    Returns
    -------
    piv : 1D Numpy array
        Pivot indices representing the permutation 
        matrix P: row i of matrix was 
        interchanged with row piv[i].

    """
    n=p.shape[0]
    piv=np.zeros(n)
    for col in range(n):
       i= np.argmax(np.abs(p[:,col]))
       piv[col]=i
       
    return piv


n=10
A=np.random.rand(n,n)
b=np.random.rand(n)

L,U,P=LU_factorize(A)

my_PA_LU_norms=np.linalg.norm(P@A-np.matmul(L,U))

print(my_PA_LU_norms)
