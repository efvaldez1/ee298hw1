#Eduardo Valdez
#2012-97976
# EE 298z HW #1

import sys
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

#python linreg.py 2 4 -1

#sys.argv[0] is the program name
sys.argv.pop(0)


x=list(map(float,sys.argv))
lenOfCoefficients=len(x)
print("Len of coefficients:",lenOfCoefficients)
x=np.reshape(x,[-1,1])

print('x: \n',x)
# 2x^2 + 4x -1


#Ax = B
#x is initially given, A could be any sample input and we solve for B

inputMatrix = np.arange(-10,10,0.01)
inputMatrix = np.reshape(inputMatrix,[-1,1])
onesColumn = np.ones(len(inputMatrix))
onesColumn = np.reshape(onesColumn,[-1,1])
#print(onesColumn)
#print(inputMatrix)

if lenOfCoefficients==2:
    '''[ x constant]'''

    inputMatrix=np.append(inputMatrix,onesColumn,axis=1)
if lenOfCoefficients==3:
    '''[ x^2 x constant]'''
    inputMatrixSquared = np.power(inputMatrix,2)
    inputMatrix=np.append(inputMatrixSquared,inputMatrix,axis=1)
    inputMatrix = np.append(inputMatrix,onesColumn,axis=1)
if lenOfCoefficients==4:
     '''[ x^3 x^2 x constant]'''
     inputMatrixCubed = np.power(inputMatrix,3)
     inputMatrixSquared = np.power(inputMatrix,2)
     inputMatrixCubed = np.append(inputMatrixCubed,inputMatrixSquared,axis=1)
     inputMatrixCubed = np.append(inputMatrixCubed,inputMatrix,axis=1)
     inputMatrix = np.append(inputMatrixCubed,onesColumn,axis=1)
print('Input Matrix!: \n',inputMatrix)

#inputMatrix * x = outputMatrix
outputMatrix=np.matmul(inputMatrix,x)
print('Output Matrix: \n',outputMatrix)

noise = np.random.uniform(-1,1,len(outputMatrix))
noise = np.reshape(noise,[-1,1])

outputMatrix=np.add(outputMatrix,noise)
print("noise :\n",noise)
print ("output with noise: \n",outputMatrix)

#set x to 0 since we have A and B. Using GDO we will find approx. of x.

#x=np.zeros(lenOfCoefficients)
x= np.random.uniform(0.0,1.0,size=lenOfCoefficients)
x=np.reshape(x,[-1,1])

print("Updated X: ",x)



tolerance= 0.1
#learningRate= 0.001
#learningRate= 0.0000001 #given 2 4 1 , it yields 1.99 3.99 -0.983
#learningRate = 0.00000001 #given 2 4 1 , it yields 2.00031866 3.99581206 -1.01305213 (takes a few seconds longer)

#seven 0's could not handle cubic functions

learningRate =  0.000000005
inputMatrixTranspose = inputMatrix.transpose()
print('A^T')
print(inputMatrixTranspose)
print('Ax')
Ax = np.matmul(inputMatrix,x)
print(Ax)


print ("LHS")
print( np.matmul(inputMatrixTranspose,Ax) )
print("RHS")
print ( np.matmul(inputMatrixTranspose,outputMatrix))
gradient = np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)
print("GD")
print(gradient)
#while (np.linalg.norm((),ord='fro')) < tolerance
#    x = x - (learningRate())

Ax = np.matmul(inputMatrix,x)

print ("Norm2 :",np.linalg.norm((np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro'))
print ("Norm2 :",np.linalg.norm((np.matmul(np.matmul(inputMatrixTranspose,inputMatrix),x) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro'))
iter = 1
while( np.linalg.norm((np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro') ) > tolerance:
    x = x - (learningRate)*(np.matmul(inputMatrixTranspose,np.matmul(inputMatrix,x)) - np.matmul(inputMatrixTranspose,outputMatrix))
    Ax = np.matmul(inputMatrix,x)
    error = np.linalg.norm(np.subtract(Ax,outputMatrix),ord='fro')
    print('iter: %d , error: %f ,learning rate: %.10f'%(iter,error,learningRate))
    iter = iter+1

print(x)



#plot
#graph of 2x^2 +4x  -1 x from -10 to 9
#vs result:
# graph of  1.999

#1.99998449
#1.4999621
#1.4999621
