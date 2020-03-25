#----------------------------------------------------------
#
# Module: bmpslib - Boundary MPS library
#
#
# History: 
# ---------
#
# 8-Oct-2019   -   Initial version.
#
#----------------------------------------------------------
#


import numpy as np
import scipy as sp
import scipy.io as sio

from scipy.sparse.linalg import eigs

from numpy.linalg import norm, svd
from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
    reshape, transpose, conj, eye, trace
from sys import exit
from copy import deepcopy


############################ CLASS MPS #################################
#                                                                      #
#                                                                      #
# Main variables:                                                      #
# ----------------                                                     #
#                                                                      #
#                                                                      #
# d - physical dimension                                               #
#                                                                      #
# N - No. of sites                                                     #
#                                                                      #
# A[] - A list of the MPS matrices. Each matrix is an array            #
#       of the shape [Dleft, d, Dright]                                #
#                                                                      #
#                                                                      #
########################################################################


class mps:
    
    #
    # --- init
    #
    
    def __init__(self,N):
        
        self.N = N
        self.A = [None]*N
        
    #--------------------------   add_site   ---------------------------
    #
    #  Sets the tensor of a site in the MPS
    #
    #  We expect an array of complex numbers of the form M[D1, d, D2]
    #
    #  The D1 should match the D2 of the previous (i.e., the last) 
    #  entry in the MPS.
    #
        
    def set_site(self, mat, i):
        
            self.A[i] = mat.copy()
            
        
    #-------------------------    mps_shape   ----------------------------
    # 
    # Returns a string with the dimensions of the matrices that make
    # up the MPS
    #
    def mps_shape(self):
        mpshape = ''
        
        for i in range(self.N):
            mpshape = mpshape + ' A_{}({} {} {}) '.\
                format(i,self.A[i].shape[0],self.A[i].shape[1], \
                self.A[i].shape[2])
                
        return mpshape
        

    #-------------------------    reduceD   ----------------------------
    # 
    # Reduce the bond dimension of the MPS using SVD from left to mid
    # and right to mid.
    #
    # If maxD is not given, then just use the maximal possible Schmidt 
    # rank.
    #
    def reduceD(self, maxD=None):
      
        #
        # Make sure that there's something to reduce. We need at least
        # 3 sites
        #
                
        if self.N<3:
            return
        
         
        #
        # First, go from left to middle
        #
        
        mid = self.N//2
        
        for i in range(mid):
            D1 = self.A[i].shape[0]
            d  = self.A[i].shape[1]
            D2 = self.A[i].shape[2]
            
            # to what dimension do we want to reduce D2
            targetD2 = d*D1 if (maxD is None) else min(maxD,d*D1)
            
            
            if D2<=targetD2:
                continue
                        
            M = self.A[i].reshape(D1*d, D2)
            
            U,S,V = svd(M, full_matrices=False)
                        
            if S.shape[0]>targetD2:
                #
                # In this case we truncate: take the largest targetD2
                # vectors
                #
                                
                S = S[0:targetD2]
                U = U[:,0:targetD2]
                V = V[0:targetD2,:]
                
                
                
                        
            M = dot(U,diag(S))
            
            self.A[i] = M.reshape(D1, d, targetD2)
            
            self.A[i+1] = tensordot(V,self.A[i+1], axes=([1],[0]))
        
            
        
        #
        #  Go from right to middle
        #
        
        for i in range(self.N-1, mid-1, -1):
            D1 = self.A[i].shape[0]
            d  = self.A[i].shape[1]
            D2 = self.A[i].shape[2]
            
            # to what dimension do we want to reduce D1
            targetD1 = d*D2 if (maxD is None) else min(maxD,d*D2)
            
            if D1<=targetD1:
                continue
            
            
            M = self.A[i].reshape(D1, d*D2)
            
            U,S,V = svd(M, full_matrices=False)

            if S.shape[0]>targetD1:
                #
                # In this case we truncate: take the largest targetD1
                # vectors
                #
                                
                S = S[0:targetD1]
                U = U[:,0:targetD1]
                V = V[0:targetD1,:]

                


                                    
            M = dot(diag(S),V)
            
            self.A[i] = M.reshape(targetD1, d, D2)
            
            self.A[i-1] = tensordot(self.A[i-1], U, axes=([2],[0]))
            
            
        return
      
      


############################ CLASS PEPS #################################
#                                                                      #
#                                                                      #
# Main variables:                                                      #
# ----------------                                                     #
#                                                                      #
# The PEPS is given as an MxN matrix with M rows and N columns         #
#                                                                      #
# M - No. of rows                                                      #                                                                     #
# N - No. of columns                                                   #
#                                                                      #
# A[][] - A double list of the PEPS matrices. Each matrix is an array  #
#         of the shape [d, D_left, D_right, D_up, D_down]              #
#         d - physical leg                                             #
#         D_left/right/up/down - virtual legs                          #
#                                                                      #
#         A[i][j] is the tensor of the i'th row and the j'th column.   #
#                                                                      # 
#                                                                      #
# Notice: Also at the boundaries every A should have exactly the  5    #
#         indices (d,Dleft,Dright,Dup,Ddown). It is expected that the  #
#         "un-needed" legs will be of dimension 1. For example, at the #
#         top-left corner A should be of the form                      #
#                         [d,1,Dright,1,Ddown]                         #
#                                                                      #
########################################################################
            
            
class peps:
    
    #
    # --- init
    #
    
    def __init__(self,M,N):
        
        self.M = M
        self.N = N
        self.A = [[None]*N for i in range(M)]
        
    #--------------------------   set_site   ---------------------------
    #
    #  Sets the tensor of a site in the PEPS
    #
    #  We expect A to be a tensor of the form:
    #                A[d, Dleft, Dright, Dup, Ddown]
    #
        
    def set_site(self, mat, i,j):
        
            self.A[i][j] = mat.copy()
            
        
    #-------------------------    peps_shape   ----------------------------
    # 
    # Returns a string with the dimensions of the matrices that make
    # up the PEPS
    #
    def peps_shape(self):
        peps_shape = ""
        
        for i in range(self.M):
            for j in range(self.N):
                if (self.A[i][j] is None):
                    peps_shape = peps_shape +  " A_{}{}(---)".format(i,j)
                else:
                    peps_shape = peps_shape +  " A_{}{}({} {} {} {}) ".\
                    format(i,j,self.A[i][j].shape[1],self.A[i][j].shape[2],\
                    self.A[i][j].shape[3],self.A[i][j].shape[4])
                
            peps_shape = peps_shape + "\n"
                
        return peps_shape
        



    #-------------------------    calc_line_MPO   ----------------------------
    # 
    # Gets a row number or a column number and return its corresponding MPO
    #
    # Parameters:
    # -----------
    # ty  -  Type of the MPO. Could be either 'row-MPS' or 'column-MPS'
    # i   -  The row/column number. Runs between 0 -> (M-1) or (N-1)
    #
  
    def calc_line_MPO(self, ty, i):
        
        if ty=='row-MPS':
            #
            # Create an MPO out of ROW i
            #
            n = self.N
            bmpo = mpo(n)
        
            for k in range(n):
                A0 = self.A[i][k]
                
                Dleft = A0.shape[1]
                Dright = A0.shape[2]
                Dup = A0.shape[3]
                Ddown = A0.shape[4]
                
                # contract the physical legs
                # Recall that in MPS the legs are: [d, Dleft, Dright, Dup, Ddown]
                #
                A = tensordot(A0,conj(A0), axes=([0],[0]))
                # 
                # resulting tensor is [Dleft,Dright,Dup,Ddown ; Dleft',Dright',Dup',Ddown']
                #                        0      1     2   3        4     5      6    7
               
                # Recall the MPO site is of the form [dup,ddown,Dleft, Dright]
                
                A = A.transpose([2,6,3,7,0,4,1,5])
                # Now its of the form [Dup, Dup', Ddown, Ddown',Dleft, Dleft', 
                #                       Dright,Dright']
        
                
                A = A.reshape([Dup*Dup, Ddown*Ddown, Dleft*Dleft, Dright*Dright])
                
                bmpo.set_site(A, k)
                
        else:
            
            #
            # Create an MPO out of COLUMN i
            #
            n = self.M
        
            bmpo = mpo(n)
        
            for k in range(n):
                A0 = self.A[k][i]

                Dleft = A0.shape[1]
                Dright = A0.shape[2]
                Dup = A0.shape[3]
                Ddown = A0.shape[4]

                
                # contract the physical legs
                A = tensordot(A0,conj(A0), axes=([0],[0]))
                # 
                # resulting tensor is [Dleft,Dright,Dup,Ddown ; Dleft',Dright',Dup',Ddown']
                #                        0      1     2   3        4     5    6    7

                
                # Recall the MPO site is of the form [dup,ddown,Dleft, Dright]
                # but here since its a vertical MPO, then:
                #
                #            up<->left  and   down<->right.
                #
                
                # So we need to transpose it to Dleft, Dleft', Dright,Dright', 
                # Dup,Dup', Ddown, Ddown'
                
                A = A.transpose([0,4,1,5,2,6,3,7])
                    
                A = A.reshape([Dleft*Dleft, Dright*Dright, Dup*Dup, Ddown*Ddown])
                
                bmpo.set_site(A, k)
            
                
                
        return bmpo
                
                
    #-------------------------    calc_bMPS   ----------------------------
    # 
    # Calculates the left/right/up/down boudary MPS
    #
    #
  
    def calc_bMPS(self, ty):
                
        if ty=='U':
            MPO = self.calc_line_MPO('row-MPS',0)
            
        elif ty=='D':
            MPO = self.calc_line_MPO('row-MPS',self.M-1)

        elif ty=='L':
            MPO = self.calc_line_MPO('column-MPS',0)
        elif ty=='R':
            MPO = self.calc_line_MPO('column-MPS',self.N-1)
        
        bmps = mps(MPO.N)
        
        for i in range(MPO.N):
            A0 = MPO.A[i]
            # A0 has the shape [dup, ddown, Dleft, Dright]
        
            if ty=='U' or ty=='L':
                d = A0.shape[1]      # ddown
                
            else:
                # ty=='D' or ty=='R'
                d = A0.shape[0]      # dup
                
            Dleft  = A0.shape[2]  # Dleft
            Dright = A0.shape[3]  # Dright
                
            A = A0.reshape([d,Dleft,Dright])
            
            A = A.transpose([1,0,2]) # make it [Dleft,d,Dright]
            
            bmps.set_site(A, i)

        return bmps
















            
                                          
            
        
        
        
############################ CLASS MPO #################################
#                                                                      #
#                                                                      #
# Main variables:                                                      #
# ----------------                                                     #
#                                                                      #
#                                                                      #
#                                                                      #
# N - No. of sites                                                     #
#                                                                      #
# Structure of the A's:  [dup,ddown,Dleft,Dright]                      #
#                                                                      #
#    dup and ddown are the physical entries and must be of the same    #
#    dimensio. dup is the input index and ddown is the output          #
#                                                                      #
########################################################################


class mpo:
    
    #
    # --- init
    #
    
    def __init__(self,N):
        
        self.N = N

        self.A = [None]*N
        
    #-------------------------    mpo_shape   ----------------------------
    # 
    # Returns a string with the dimensions of the matrices that make
    # up the MPO
    #
    def mpo_shape(self):
        mpo_shape = ''
        
        for i in range(self.N):
            mpo_shape = mpo_shape + ' A_{}({} {}; {} {}) '.\
                format(i,self.A[i].shape[0],self.A[i].shape[1], \
                    self.A[i].shape[2],self.A[i].shape[3])
                
        return mpo_shape
        
    #--------------------------   set_site   ---------------------------
    #
    #  Sets the tensor of a site in the MPS
    #
    #  We expect an array of complex numbers of the form 
    #  M[D1, dup, ddown, D2]
    #
    #  The D1 should match the D2 of the previous (i.e., the last) 
    #  entry in the MPO
    #
        
    def set_site(self, mat, i):
        
            self.A[i] = mat.copy()
            
                
#                
# **********************************************************************
# 
#                         F U N C T I O N S
#
# **********************************************************************
#     
        

# ----------------------------------------------------------------------
#  applyMPO
#
#  Apply the MPO op to the MPS M. If OverWrite=True then result is 
#  written to M itself. Default: OverWrite=False
#
#  Parameters:
#  op       - The MPO
#  M        - the MPS
#  i1       - An optional parameter specifying the initial location on M
#             where op start acting.
#  cont_leg - which leg of the MPO to contract - could be 'U' (up)
#             or 'D' (Down).
#
#

def applyMPO(op, M, i1=0, cont_leg='U'):
    
  
    newM = mps(M.N)
  
    for i in range(i1, i1+op.N):
        
        MD1 = M.A[i].shape[0]
        Md  = M.A[i].shape[1]
        MD2 = M.A[i].shape[2]
        
        d_up   = op.A[i-i1].shape[0]
        d_down = op.A[i-i1].shape[1]
        opD1   = op.A[i-i1].shape[2]
        opD2   = op.A[i-i1].shape[3]
        
        #
        # Recall that the MPO legs order is [d_up, d_down, D_left, D_right]
        # while the MPS element legs are     [D_left, d, Dright]
        
        if cont_leg=='U':
            newA = tensordot(M.A[i], op.A[i-i1], axes=([1],[0]))
            # newA has the legs [MD1, MD2, d_down, opD1, opD2]
            #                     0    1     2       3    4
            
            # transpose it to: [(MD1, opD1), d_down, (MD2, opD2)]
            
            newA = newA.transpose([0,3,2,1,4])
            newA = newA.reshape( [MD1*opD1, d_down, MD2*opD2])
        else:
            #
            # So cont_leg = 'D'
            #
            
            newA = tensordot(M.A[i], op.A[i-i1], axes=([1],[1]))
            # newA has the legs [MD1, MD2, d_up, opD1, opD2]
            #                     0    1     2       3    4
            
            # transpose it to: [(MD1, opD1), d_up, (MD2, opD2)]
            
            newA = newA.transpose([0,3,2,1,4])
            newA = newA.reshape( [MD1*opD1, d_up, MD2*opD2])
            
        newM.set_site(newA, i)
            
        
    return newM
        

# ---------------------------------------------------------
#  enlargePEPS
#
#  Add trivial rows and columns that surround the original PEPS.
#  This is useful when calculating properties on the boundary on the
#  original PEPS. If these sites are now in the bulk then it might be
#  easier to do the calculations.
#

def enlargePEPS(p):
    
    M = p.M
    N = p.N
    
    newp = peps(M+2, N+2)
    
    trivialA = array([1])
    trivialA = trivialA.reshape([1,1,1,1,1])
    
    for i in range(N+2):
        newp.A[0][i] = trivialA.copy()
        newp.A[M+1][i] = trivialA.copy()
        
    for i in range(M+2):
        newp.A[i][0] = trivialA.copy()
        newp.A[i][N+1] = trivialA.copy()
        
    for i in range(M):
        for j in range(N):
            A = p.A[i][j]
            newp.A[i+1][j+1] = A.copy()
            
            
    return newp
            


# ---------------------------------------------------------
#  updateCOLeft 
#
#  Update the contraction of an MPO with 2 MPSs from left to right.
#
#  If C is not empty then its a 3 legs tensor:
#  [Da,Do,Db]
#  

def updateCOLeft(C, A, Op, B):
    
    
    if C is None:
        #
        # When C=None we are on the left most side. So we create COLeft
        # from scratch
        #
        
        # A is  [1, dup, D2a]
        # Op is [up, ddown, 1, D2o]
        # B is  [1, ddown, D2b]
        
        C1 = tensordot(A[0,:,:], Op[:,:,0,:], axes=([0],[0]))
        
        # C legs are: [D2a, ddown, D2o]
        
        C1 = tensordot(C1, B[0,:,:], axes=([1],[0]))
        
        # result is: [D2a,D2o,D2b]
        
        return C1
    
    
    # C is given as  [D1a, D1o, D1b]
    # A is given as  [D1a, dup, D2a]
    # Op is given as [dup, down, D1o, D2o]
    # B is given as  [D1b, ddown, D2b]
    
    C1 = tensordot(C,Op, axes=([1],[2]))
    # C1: [D1a,D1b,dup, ddown, D2o]
    
    C1 = tensordot(C1, A, axes=([0,2],[0,1]))
    # C1: [D1b, ddown, D2o, D2a]
    
    C1 = tensordot(C1, B, axes=([0,1], [0,1]))
        
    # C1: [D2o, D2a, D2b]
    
    C1 = C1.transpose([1,0,2])
    
    
    return C1
    




# ---------------------------------------------------------
#  updateCORight
#
#  Update the contraction of an MPO and two MPSs from right to left
#
#  If C is not empty then its a 3 legs tensor:
#  [Da,Do,Db]
#  

def updateCORight(C, A, Op, B):
    
    
    if C is None:
        #
        # When C=None we are on the left most side. So we create COLeft
        # from scratch
        #
        
        # A is  [D1a, dup, 1]
        # Op is [dup, ddown, D1o, 1]
        # B is  [D1b, ddown, 1]
        
        C1 = tensordot(A[:,:,0], Op[:,:,:,0], axes=([1],[0]))
        
        # C1 legs are: [D1a, ddown, D1o]
        
        C1 = tensordot(C1, B[:,:,0], axes=([1],[1]))
        
        # result is: [D1a,D1o,D1b]
        
        return C1
    
    
    # C is given as  [D2a, D2o, D2b]
    # A is given as  [D1a, dup, D2a]
    # Op is given as [dup, ddown, D1o, D2o]
    # B is given as  [D1b, ddown, D2b]
    
    C1 = tensordot(C,Op, axes=([1],[3]))
    # C1: [D2a,D2b,dup, ddown, D1o]
    
    C1 = tensordot(C1, A, axes=([0,2],[2,1]))
    # C1: [D2b, ddown, D1o, D1a]
    
    C1 = tensordot(C1, B, axes=([0,1], [2,1]))
        
    # C1: [D1o, D1a, D1b]
    
    C1 = C1.transpose([1,0,2])
    
    
    return C1



# ---------------------------------------------------------
#  updateCLeft 
#
#  Update the contraction of two MPSs from left to right.
#
#  If C is not empty then its a 2 legs tensor:
#  [Da,Db]
#  

def updateCLeft(C, A, B, conjB=False):
    
    
    if C is None:
        #
        # When C=None we are on the left most side. So we create CLeft
        # from scratch
        #
        
        # A is  [1, dup, D2a]
        # B is  [1, ddown, D2b]
        
        if conjB:
            C1 = tensordot(A[0,:,:], conj(B[0,:,:]), axes=([0],[0]))
        else:
            C1 = tensordot(A[0,:,:], B[0,:,:], axes=([0],[0]))
            
        # C1 legs are: [Da2, Db2]
        
        
        return C1
    
    
    # C is given as  [D1a, D1b]
    # A is given as  [D1a, dup, D2a]
    # B is given as  [D1b, ddown, D2b]
    
    C1 = tensordot(C,A, axes=([0],[0]))
    # C1: [D1b,dup,Da2]
        
    if conjB:
        C1 = tensordot(C1, conj(B), axes=([0,1], [0,1]))        
    else:
        C1 = tensordot(C1, B, axes=([0,1], [0,1]))
        
    # C1: [D2a, D2b]
        
    return C1
    


# ---------------------------------------------------------
#  mps_inner_product - return the inner product <A|B>
#

def mps_inner_prodcut(A,B,conjB=False):
    
    leftC = None
    
    for i in range(A.N):
        
        leftC = updateCLeft(leftC,A.A[i],B.A[i], conjB)
        
    return leftC[0,0]
        
# ---------------------------------------------------------
#  mps_sandwitch - return the expression <A|O|B>
#  for MPSs A,B and an MPO O
#

def mps_sandwitch(A,Op,B,conjB=False):
    
    leftCO = None
    
    for i in range(A.N):
        
        if conjB:
            leftCO = updateCOLeft(leftCO,A.A[i],Op.A[i],conj(B.A[i]))
        else:
            leftCO = updateCOLeft(leftCO,A.A[i],Op.A[i],B.A[i])
        
    return leftCO[0,0,0]
        


# ---------------------------------------------------------
#
# calculate_2RDM_from_a_list - get 2 boundary MPSs (one from above and
# one from below, together with a list of PEPS matrices that 
# go between them and outputs the list of 2-local RDM from the bulk
# of the line.
#

def calculate_2RDM_from_a_line(bmpsU, bmpsD, A):
    
    #
    # first step: calculate the MPO made from the A list.
    #
    
    N = len(A)
    
    oplist=[]
    for i in range(N):
        A0 = A[i]
        
        Dleft = A0.shape[1]
        Dright = A0.shape[2]
        Dup = A0.shape[3]
        Ddown = A0.shape[4]
        
        # contract the physical legs
        # Recall that in MPS the legs are: [d, Dleft, Dright, Dup, Ddown]
        #
        Aop = tensordot(A0,conj(A0), axes=([0],[0]))
        # 
        # resulting tensor is [Dleft,Dright,Dup,Ddown ; Dleft',Dright',Dup',Ddown']
        #                        0      1     2   3        4     5      6    7
       
        # Recall the MPO site is of the form [dup,ddown,Dleft, Dright]
        
        Aop = Aop.transpose([2,6,3,7,0,4,1,5])
        # Now its of the form [Dup, Dup', Ddown, Ddown',Dleft, Dleft', 
        #                       Dright,Dright']
        
        Aop = Aop.reshape([Dup*Dup, Ddown*Ddown, Dleft*Dleft, Dright*Dright])
        
        oplist.append(Aop)
        
    
    rhoL = []
    
    #
    # Now go over all sites in the bulk and calculate the (j1, j1+1) RDM.
    #
    # To do that we need two tensors which are the 
    # contraction of the 3 layers:
    #
    # 1. CLO - contraction from 0 to j1-1
    # 2. CRO - contraction from N-1 to j1+2
    #
    # Once these are found, rho can be calculated from them + the 
    # bmpsU, bmpsD tensors of j1,j1+1 as well as the PEPS tensors at 
    # j1,j1+1
    #
    
    
    CLO=None
    for j1 in range(1,N-2):        
        CLO = updateCOLeft(CLO, bmpsU.A[j1-1], oplist[j1-1], bmpsD.A[j1-1])
        # CLO legs: Da1, Do1, Db1


        CRO=None
        for j in range(N-1, j1+1, -1):
            CRO = updateCORight(CRO, bmpsU.A[j], oplist[j], bmpsD.A[j])

        # CRO legs: Da3, Do3, Db3    
        
        AupL = bmpsU.A[j1]    # legs: Da1, ddown1, Da2
        AupR = bmpsU.A[j1+1]  # legs: Da2, ddown2, Da3

        AdownL = bmpsD.A[j1]   # legs: Db1, dup1, Db2
        AdownR = bmpsD.A[j1+1] # legs: Db2, dup2, Db3

        CLO1 = tensordot(CLO, AupL, axes=([0], [0]))  # legs: Do1,Db1,ddown1,Da2
        CLO1 = tensordot(CLO1, AdownL, axes=([1],[0])) # legs: Do1,ddown1, Da2, dup1, Db2
        CLO1 = CLO1.transpose([2,0,4,3,1]) # legs: Da2, Do1, Db2, dup1, ddown1
                                         #        0    1    2      3     4
                                         
        Ai = A[j1]     # legs: di, Dileft, Diright, Diup, Didown

        di      = Ai.shape[0]
        Dileft  = Ai.shape[1]
        Diright = Ai.shape[2]
        Diup    = Ai.shape[3]
        Didown  = Ai.shape[4]

        AiAi = tensordot(Ai,conj(Ai),0) 
        AiAi = AiAi.transpose([0,5,1,6,2,7,3,8,4,9])
        AiAi = AiAi.reshape([di,di,Dileft**2, Diright**2, Diup**2, Didown**2])

        CLO1 = tensordot(CLO1, AiAi, axes=([1,4,3],[2,4,5]))
        # CLO legs: Da2, Db2, di, c-di, Diright  


        CRO = tensordot(CRO, AupR, axes=([0],[2])) # legs: Do3, Db3, Da2, ddown2
        CRO = tensordot(CRO, AdownR, axes=([1],[2])) # legs: Do3, Da2, ddown2, Db2, dup2
        CRO = CRO.transpose([1,0,3,4,2])  # legs: Da2, Do3, Db2, dup2, ddown2

        Aj = A[j1+1]  # legs: dj, Djleft, Djright, Djup, Djdown

        dj      = Aj.shape[0]
        Djleft  = Aj.shape[1]
        Djright = Aj.shape[2]
        Djup    = Aj.shape[3]
        Djdown  = Aj.shape[4]

        AjAj = tensordot(Aj,conj(Aj),0) 
        AjAj = AjAj.transpose([0,5,1,6,2,7,3,8,4,9])
        AjAj = AjAj.reshape([dj,dj,Djleft**2, Djright**2, Djup**2, Djdown**2])
        #                     0  1    2         3           4        5

        CRO = tensordot(CRO, AjAj, axes=([1,4,3],[3,4,5]))
        # CRO legs: Da2,Db2, dj,c-dj, Djleft


        rho = tensordot(CLO1, CRO, axes=([0,1,4], [0,1,4]))
        # rho legs: di,c-di),dj,c-dj)
        
        rho = rho/trace(trace(rho))
        
        rhoL.append(rho)
    
    
    return rhoL


# ---------------------------------------------------------
#
#  Given a PEPS and a truncation bond dimension Dp, this function
#  uses the boundary-MPS method to calculate the 2-body RDM of every
#  link in the MPS. The output is a list of all such RDMs, starting
#  from the horizontal RDMS: 
#    [(0,0), (0,1)] , [(0,1),(0,2)], ... , [(0,N-2),(0,N-1)],
#    [(1,0), (1,1)] , [(1,1),(1,2)], ... , [(1,N-2),(1,N-1)], 
#    ...
#
#  And the the vertical RDMS:
#
#    [(0,0), (1,0)], [(1,0), (2,0)], ... , [(M-2,0),(M-1,0)],
#    [(0,1), (1,1)], [(1,1), (2,1)], ... , [(M-2,1),(M-1,1)],
#    ...
#
#  Notice that there are (M-1)N horizontal RDMs and M(N-1) vertical RDMs.
#
#  Every 2-body RDM is a tensor of the form:
# 
#                   rho_{alpha_1,beta_1; alpha_2,beta_2}
#
#  where the alpha_1,alpha_2 correspond to the ket and beta_1,beta_2 
#  correspond to the bra (i.e, the complex-conjugated part).
#
#  Every rho is normalized to have a trace=1.
#
#

def calculate_PEPS_2RDM(p0, Dp):
    
    
    
    rhoL = []
    
    #
    # We pass to an enlarged PEPS, by adding trivial rows at the 
    # top/bottom and trivial columns at left/right. This way the 
    # bulk RDMS of the enlarged PEPS 
    #
    p = enlargePEPS(p0)
   
    #
    # =======  Calculate HORIZONTAL 2-body RDMs ==========
    #
    
       
    #
    # Go over all rows and for each row calculate the boundary-MPS 
    # above it (bmpsU) and the one below it (bmpsD)
    #
    
    bmpsU = p.calc_bMPS('U')
    bmpsU.reduceD(Dp)
        
    for i1 in range(1,p.M-1):
        
        bmpsD = p.calc_bMPS('D')
        bmpsD.reduceD(Dp)
        
        for i in range(p.M-2,i1,-1):
            op = p.calc_line_MPO('row-MPS',i)
            bmpsD = applyMPO(op,bmpsD,cont_leg='D')
            bmpsD.reduceD(Dp)
            
        
        AList = []
        for j in range(p.N):
            AList.append(p.A[i1][j])
            
        rhoL = rhoL + calculate_2RDM_from_a_line(bmpsU, bmpsD, AList)
        
        #
        # Updating the bmpsU
        # 
        op = p.calc_line_MPO('row-MPS',i1)
        bmpsU = applyMPO(op,bmpsU,cont_leg='U')
        bmpsU.reduceD(Dp)
        
        

    #
    # =======  Calculate VERTICAL 2-body RDMs ==========
    #
    
       
    #
    # Go over all columns from left to right and for each column 
    # calculate the boundary-MPS from its left (bmpsL) and from
    # its right (bmpsR)
    #
    
    bmpsL = p.calc_bMPS('L')
    bmpsL.reduceD(Dp)
        
    for j1 in range(1,p.N-1):
        

        bmpsR = p.calc_bMPS('R')
        bmpsR.reduceD(Dp)
        
        for j in range(p.N-2,j1,-1):
            op = p.calc_line_MPO('column-MPS',j)
            bmpsR = applyMPO(op,bmpsR,cont_leg='D')
            bmpsR.reduceD(Dp)
            
        
        AList = []
        for i in range(p.M):
            A = p.A[i][j1]
            #
            # transpose A's indices so that the left/right indices
            # are moved to up/down to match the requirements of the
            # calculate_2RDM_from_a_line function requirements.
            #
            
            A = A.transpose([0,3,4,1,2])
            
            AList.append(A)
            
        rhoL = rhoL + calculate_2RDM_from_a_line(bmpsL, bmpsR, AList)
        
        #
        # Updating the bmpsL
        # 
        op = p.calc_line_MPO('column-MPS',j1)
        bmpsL = applyMPO(op,bmpsL,cont_leg='U')
        bmpsL.reduceD(Dp)
        




    return rhoL

        

