import numpy as np 
import theano.tensor as TT
from theano import function

matrix_A = np.array([[[0.2,0.3,0.5,0.4,0.6,0.7],[0.1,0.2,0.3,0.4,0.5,0.6],[0.1,0.2,0.3,0.4,0.5,0.6],[0.1,0.2,0.3,0.4,0.5,0.6]]])

matrix_A = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[101,102,103,104],[105,106,107,108],[109,110,111,112]]])
matrix_A.shape #(2, 3, 4)


matrix_T  = np.array([[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.5,0.6,0.7,0.8],[0.2,0.4,0.5,0.1]])
matrix_T.shape #(1, 4, 4)

Points = np.array([[0,1,2], [1,2,3]])

A = TT.dtensor3()
T = TT.dmatrix()
Point = TT.dmatrix()

#out = A[0][0] + T[0][0]+Point[0]
out = A[0][0][Point[0][0]] + A[0][0][Point[0][1]]+T[Point[0][1]]
f = function([A, T, Point], out)

print f(matrix_A,matrix_T, Points)

i=0
matrix_A[0][i][Points[0][i]]
i=1
matrix_A[0][i][Points[0][i]]+matrix_T[Points[0][i-1]][Points[0][i]]
i=2
matrix_A[0][i][Points[0][i]]+matrix_T[Points[0][i-1]][Points[0][i]]

