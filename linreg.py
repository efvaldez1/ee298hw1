#Eduardo Valdez
#2012-97976
#EE 298z HW-1 Gradient Descent Operation

import sys
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

#python linreg.py 2 4 -1
#sys.argv[0] is the program name and all the remaining are the coefficients of the polynomial
sys.argv.pop(0)
coefficients=list(map(float,sys.argv))
x=coefficients
lenOfCoefficients=len(x)
x=np.reshape(x,[-1,1])
#print('x: \n',x)


#Ax = B
#x is initially given, A could be any sample input and we solve for B
#As told by Dr.Atienza, range(-10,10) is fine
inputMatrix = np.arange(-10,10,0.1)
sample=inputMatrix
lenOfInput=len(inputMatrix)
inputMatrix = np.reshape(inputMatrix,[-1,1])
onesColumn = np.ones(len(inputMatrix))
onesColumn = np.reshape(onesColumn,[-1,1])


#We could set any constant  for x^0  as told by Dr.Atienza, however I stick with his example of using 1.
if lenOfCoefficients==2:
    ''' [ [ x constant]
                ...
                ...    ]'''
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


#inputMatrix * x = outputMatrix
outputMatrix=np.matmul(inputMatrix,x)
#print('Output Matrix: \n',outputMatrix)

#We add noise from -1.0 to 1.0 from Normal Distribution as suggested by Dr.Atienza
noise = np.random.uniform(-1.0,1.0,len(outputMatrix))
noise = np.reshape(noise,[-1,1])
outputMatrixOriginal = outputMatrix
outputMatrixWithNoise=np.add(outputMatrix,noise)

outputMatrix=outputMatrixWithNoise
#print("noise :\n",noise)
#print ("output with noise: \n",outputMatrix)

#Disregard previous values of x
#Initialize x to random weights
# since we have A and B. Using GDO we will find approx. of x.
#x=np.zeros(lenOfCoefficients)
x= np.random.uniform(0.0,1.0,size=lenOfCoefficients)
x=np.reshape(x,[-1,1])
print("Updated X: ",x)



tolerance= 0.1

#learningRate= 0.0000001                    #given 2 4 1 , it yields 1.99 3.99 -0.983
#learningRate = 0.00000001                  #given 2 4 1 , it yields 2.00031866 3.99581206 -1.01305213 (takes a few seconds longer)
learningRate =  0.0001
inputMatrixTranspose = inputMatrix.transpose()

#print('A^T')
#print(inputMatrixTranspose)
#print('Ax')
Ax = np.matmul(inputMatrix,x)
#print(Ax)


#print ("LHS")
#print( np.matmul(inputMatrixTranspose,Ax) )
#print("RHS")
#print ( np.matmul(inputMatrixTranspose,outputMatrix))
gradient = np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)
#print("GD")
#print(gradient)
#while (np.linalg.norm((),ord='fro')) < tolerance
#    x = x - (learningRate())

Ax = np.matmul(inputMatrix,x)

#print ("Norm2 :",np.linalg.norm((np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro'))
#print ("Norm2 :",np.linalg.norm((np.matmul(np.matmul(inputMatrixTranspose,inputMatrix),x) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro'))


iter = 1

'''
Reference for BOLD DRIVER ALGORITHM
http://www.willamette.edu/~gorr/classes/cs449/momrate.html
http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/
I used a slighty modified version, such that:
 If your error rate was reduced since the last iteration, you can try increasing the learning rate by 5%.
 If your error rate was actually increased (meaning that you skipped the optimal point) decrease the learning rate by 50%
'''
previousGradient = np.linalg.norm((np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro')
#print(previousGradient)
while   ( np.linalg.norm((np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro') ) > tolerance:
      currentGradient=( np.linalg.norm((np.matmul(inputMatrixTranspose,Ax) - np.matmul(inputMatrixTranspose,outputMatrix)),ord='fro') )
      if(previousGradient > currentGradient):
          learningRate=learningRate + (learningRate*0.05)
      if(currentGradient > previousGradient):
          learningRate=learningRate - (learningRate*0.5)
      previousGradient=currentGradient

      x = x - (learningRate)*(np.matmul(inputMatrixTranspose,np.matmul(inputMatrix,x)) - np.matmul(inputMatrixTranspose,outputMatrix))
      Ax = np.matmul(inputMatrix,x)

      error = np.linalg.norm(np.subtract(Ax,outputMatrix),ord='fro')
      print('x: ',x.flatten(),' iter: %d , error: %20f ,learning rate: %.20f \n'%(iter,error,learningRate))
      iter = iter+1

#print(x)
print('x after GDO: ',x.flatten())


#PLOTS
#graph of input and output (with added uniform noise ) i.e 2x^2 +4x  -1  from -10 to 10 at 0.1 interval
#vs result of graph of input and output (computer using x or 'weights' after GDO):
# graph input and out pusing using weights of  1.999 1.99998449 1.4999621 1.4999621 after running GDO



# def f(x,coeffs):
#    size = len(coeffs)
#    if size == 2:
#        return (coeffs[0]*x)+(coeffs[1])
#    if size == 3:
#        return (coeffs[0]*x**2)+(coeffs[1]*x)+(coeffs[2])
#    if size == 4:
#        return (coeffs[0]*x**3)+(coeffs[1]*x**2)+(coeffs[2]*x)+(coeffs[3])
# print('Output Original: ')
# print(outputMatrixOriginal.flatten())
# print('Noise :')
# print(noise.flatten())
# print("Output of original (with Uniform Noise)" )
# print(outputMatrixWithNoise.flatten())
#
# print("Output using x after GDO")
# print(f(sample,x.flatten()))
#
#
# plt.plot(sample, outputMatrixWithNoise, 'b-',  sample, f(sample,x.flatten()), 'r--')
# plt.show()
