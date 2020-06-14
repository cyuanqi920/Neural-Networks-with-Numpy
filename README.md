# deepnn

Yo... this will be a quick tutorial on how to use the code.
My code is made based on the numpy module.
Firstly save the code into a file named file.py

import file
import numpy as np

train_x = np.array([[0,1,2,3,4,5,6,7,8,9,  10,11,12,13,14,15,16,17,18,19],
                    [1,2,3,4,5,6,7,8,9,10, 9, 10,11,12,13,14,15,16,17,18]])
train_y = np.array([[1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1]])
n = [2,5,10,5,2]
reg_param = 1
learning_param = 0.025

model = file.DeepNN(train_x, train_y, n, reg_param, learning_param)


