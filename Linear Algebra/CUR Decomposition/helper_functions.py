# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:16:52 2023

@author: anjan
"""


import numpy as np


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

def swap_cols(A,p,q):
    """
    Parameters
    ----------
    A: A numpy.ndarry matrix of any dimensions
    
    p,q : integers
        The indices of two columns of the matrix.

    Returns
    -------
    numpy.ndarry matrix A with col p swapped with col q.

    """
    A[:,[p,q]]=A[:,[q,p]]
    
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
    swap_rows(U[:,k:],k,max_index)
    swap_rows(P,k,max_index)
    swap_rows(L[:,:k],k,max_index)
    
    return L,P,U

def complete_pivot(L,P,U,Q,k,max_row_i,max_col_i):
    """
    
    Parameters
    ----------
    max_row_i: integer
        The row index of the maximum of the absolutes
        in the submatrix A(k:n,k:n)
    max_col_i: integer
        The col index of the maximum of the absolutes
        in the submatrix A(k:n,k:n)    
     L : square matrix
        A Lower Triangular Matrix
        in the intermediate stage of a Gaussian Elimination
    P : square matrix
        The row permuation matrix
    Q : square matrix
        The column permutation matrix
    U : square matrix
        The matrix in the intermediate stage of Gaussian Elimination
        that would eventually return an Upper triangular matrix
    k : integer
        The row column index of the pivot of the matrix U

    Returns
    -------
    numpy.ndarry matrices L,P,U with rows pivoted

    """
    swap_rows(U[:,k:],k,max_row_i)
    swap_cols(U[:,:],k,max_col_i)
    swap_rows(P,k,max_row_i)
    swap_rows(L[:,:k],k,max_row_i)
    #L[:k,:]=swap_cols(L[:k,:],k,max_col_i)
    swap_cols(Q, k, max_col_i)
    
    return L,P,U,Q


def rook_pivot(L,P,U,Q,k,i,method):
    """
    
    Parameters
    ----------
    i: integer
        The index of the maximum of the absolutes
        in the pivot column and row   
     L : square matrix
        A Lower Triangular Matrix
        in the intermediate stage of a Gaussian Elimination
    P : square matrix
        The row permuation matrix
    Q : square matrix
        The column permutation matrix
    U : square matrix
        The matrix in the intermediate stage of Gaussian Elimination
        that would eventually return an Upper triangular matrix
    k : integer
        The row column index of the pivot of the matrix U
    method : string
        Indicates whether the row or column matrix contains the maximum 
        absolute value

    Returns
    -------
    numpy.ndarry matrices L,P,U with rows pivoted

    """
    if method=='column':
         return complete_pivot(L, P, U, Q, k, i, k)
    elif method=='row':
        return complete_pivot(L, P, U, Q, k, k, i)





def GE_step(L,U,k,j):
    # This function performs the Gaussian
    # Elimination step for row j and column k
    
    L[j,k]=U[j,k]/U[k,k]
    U[j, k: ]=U[j,k:]-L[j,k]*U[k,k:]
    return L,U


def LU_factorize_complete(A):

    
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
    
    eps=1.101020e-60
    n=A.shape[0]
    A=A.astype(float)
    L=np.identity(n)
    P=np.identity(n)
    Q=np.identity(n)
    U=A.copy()
    #pivots=[]
    
    for k in range(n):
        max_tuple=np.unravel_index(np.argmax(np.abs(U[k:, k:])), U[k:, k:].shape)
        max_row_i,max_col_i =  max_tuple[0]+k,max_tuple[1]+k
        
        if abs(U[max_row_i,max_col_i])<eps:
           return ("Pivot {} less than tolerance"
                   .format(U[max_row_i,max_col_i])),None,None,None
   
        if (max_row_i,max_col_i) != (k,k):
              L,P,U,Q=complete_pivot(L,P,U,Q,k,max_row_i,max_col_i)

        for j in range(k+1,n):
           L,U=GE_step(L,U,k,j)
           
    return L,U,P,Q


def LU_factorize_rook(A):

    
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
    
    eps=1.0e-60
    n=A.shape[0]
    A=A.astype(float)
    L=np.identity(n)
    P=np.identity(n)
    Q=np.identity(n)
    U=A.copy()
    #pivots=[]
    
    for k in range(n):
        max_col_i=k + np.argmax(np.abs(U[k:, k]))
        max_row_i=k+np.argmax(np.abs(U[k, k:]))
        
        if abs(U[k,max_col_i])<abs(U[max_row_i,k]):
            i=max_row_i
            method='row'
            if abs(U[i,k])<eps:
               return ("Pivot {} less than tolerance"
                       .format(U[i,k])),None,None,None
        else:
            method='column'
            i=max_col_i
            if abs(U[k,i])<eps:
               return ("Pivot {} less than tolerance"
                       .format(U[i,k])),None,None,None
        if i != k:
              L,P,U,Q=rook_pivot(L,P,U,Q,k,i,method=method)

        for j in range(k+1,n):
           L,U=GE_step(L,U,k,j)
           
    return L,U,P,Q


def LU_factorize_partial(A):
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
    
    
    eps=1.0e-60
    n=A.shape[0]
    A=A.astype(float)
    L=np.identity(n)
    P=np.identity(n)
    Q=np.identity(n)
    U=A.copy()
    pivots=[]
    
    for k in range(n):
        max_index = k + np.argmax(np.abs(U[k:, k]))
       
        if abs(U[max_index,k])>eps:
            pivots.append(U[max_index,k])
            
        else:
            return ("Pivot {} less than tolerance"
                    .format(U[max_index,k])),None,None,None
   
        if max_index != k:
              L,P,U=partial_pivot(L,P,U,k,max_index)

        for j in range(k+1,n):
           L,U=GE_step(L,U,k,j)
           
    return L,U,P,Q

def substitution(L,U,P,Q,b):
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
    
    
    #Solve Lz=Pb using forward substitution
    
    b_=np.matmul(P,b)
    z=np.zeros(n)
    
    for i in range(len(z)):
        if i==0:
            z[i]=b_[0]
        else:
           z[i]=(b_[i]-np.matmul(L[i,:i],z[:i]))/(L[i,i])
    
    
    #Solve Uy=z using backward substitution

    y=np.zeros(n)
    for j in range(n-1,-1,-1):
        if j==n-1:
            y[j]=z[j]/U[j,j]
        else:
            y[j]=(z[j]-np.matmul(U[j,j+1:],y[j+1:]))/U[j,j]
    
    return Q@y


def max_diag_index(A,j):
    """
    Finds absolute maximum number in the subdiagonal of a matrix
    """
    
    n=A.shape[0]
    max_d=A[j,j]
    max_d_i=j
    for i in range(j,n):
        if abs(max_d)>abs(A[i,i]):
            max_d=A[i,i]
            max_d_i=i
    
    return max_d_i

def Cholesky_pivot(A,G,P,j,max_index):
    """
    Pivots the current pivot element with the desired element in A,G,P
    """
    
    swap_cols(A[:,:], j, max_index)
    swap_rows(A[:,:], j, max_index)
    swap_cols(G[:,:], j, max_index)
    swap_rows(G[:,:], j, max_index)
    swap_rows(P,j,max_index)
    
    return A,G,P



def my_cholesky(A):
    """
    Parameters
    ----------
    A : 2D Numpy Array
        A positive semi definite symmetric Matrix

    Returns
    -------
    G : The Cholesky Factor such that GG^T = A .
        
    """
    
    eps=1.0e-60
    n=A.shape[0]
    A=A.astype(float)
    G=np.zeros((n,n))
    P=np.identity(n)
    
    for j in range(n):
        max_index=max_diag_index(A,j)
                
        if abs(A[max_index,max_index])<eps:
            return ("Pivot {} less than tolerance"
                    .format(A[max_index,max_index])),None
        
        if max_index != j:
              A,G,P=Cholesky_pivot(A,G,P,j,max_index)
            
        v=A[j:,j]
        
      

        if j>0:
            for k in range(j):
                
                v=v-G[j,k]*G[j:,k]

            
        G[j:,j]=v/np.sqrt(v[0])

    
    
    return G,P


    

# n=4

# A=np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         A[i,j]=1/(i+j+1)    

# b=np.random.rand(n)

# L,U,P,Q=LU_factorize_complete(A)
# L1,U1,P1,Q1=LU_factorize_partial(A)
# L2,U2,P2,Q2=LU_factorize_rook(A)

# G,P0=my_cholesky(A)

# from scipy.linalg import cho_factor, cho_solve, cholesky

# L = cholesky(A, lower=True)
# print('Cholesky (PAP^T-GG^T) Norm:',np.linalg.norm(P0@A@(P0.transpose())-G@(G.transpose())))

# if U is not None:
#     print('Complete Pivot (PAQ-LU) Norm:',np.linalg.norm(P@A@Q-L@U))
#     x=substitution(L,U,P,Q,b)
#     print('Complete Pivot (Ax-b) Norm:',np.linalg.norm(A@x-b) )
#     print('========================================')
# else:
#     print(L)
    
# if U1 is not None:
#     print('Rook Pivot (PAQ-LU) Norm:', np.linalg.norm(P2@A@Q2-L2@U2))
#     x2=substitution(L2,U2,P2,Q2,b)
#     print('Rook Pivot (Ax-b) Norm:',np.linalg.norm(A@x2-b) )
    
#     print('========================================')
# else:
#     print(L1)
    
# if U2 is not None:
#     print('Partial Pivot (PAQ-LU) Norm: ', np.linalg.norm(P1@A-L1@U1))
#     x1=substitution(L1,U1,P1,Q1,b)
#     print('Partial Pivot (Ax-b) Norm:',np.linalg.norm(A@x1-b) )
    
#     print('========================================')
# else:
#     print(L2)


def cholesky_decomposition(matrix):
    """Performs matrix decomposition using the Cholesky method.
    For more information, `check here <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_
    Parameters
    ----------
    matrix : list of lists
        A matrix filled with numbers (no matter int or float) to perform Cholesky decomposition on.
  
    Returns
    -------
    L : list of lists
        The lower triangular matrix coming from the decomposition.
    """

    d = len(matrix)
    # Initializing an nxn matrix of zeros
    L = [[0 for i in range(d)] for j in range(d)]

    # Initializing the first element in the matrix
    L[0][0] = (matrix[0][0])**0.5

    # Initializing the first column of the matrix
    for i in range(1, d):
        L[i][0] = (matrix[0][i]) / (L[0][0])
    
    # Filling-in elsewhere
    for i in range(1, d):
        for j in range(1, i+1):
            # Filling the main diagonal
            if i == j:
                L[i][j] = (matrix[i][j] - sum((L[i][k]**2) for k in range(0, i)))**0.5
            
            # Filling below the main diagonal
            else:
                L[i][j] = (1 / L[j][j]) * (matrix[i][j] - sum(L[i][k]*L[j][k] for k in range(0, min(i,j))))
    
    return 
    
n=12

A=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i,j]=1/(i+j+1)    

b=np.random.rand(n)

G=cholesky_decomposition(A)
