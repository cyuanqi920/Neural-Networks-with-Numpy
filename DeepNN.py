import numpy as np


class DeepNN:

    def __init__(self, train_x, train_y, n, regularisation_parameter, learning_parameter):
        self.x = train_x
        self.y = train_y
        self.n = n
        self.L = len(self.n) - 1
        self.h = None
        try:
            self.m = np.shape(self.y)[1]
        except:
            self.m = 1
        self.reg = regularisation_parameter
        self.alpha = learning_parameter
        self.A = [None] * ((self.L)+1)
        self.W = [None] * ((self.L)+1)
        self.b = [None] * ((self.L)+1)
        self.Z = [None] * ((self.L)+1)
        self.dA = [None] * ((self.L)+1)
        self.dW = [None] * ((self.L)+1)
        self.db = [None] * ((self.L)+1)
        self.dZ = [None] * ((self.L)+1)

    def relu(Z):
        return np.maximum(0,Z)

    def sigmoid(Z):
        a = 1.0 + np.exp(-Z)
        return 1.0 / a

    def reshape(data):
        def average(data):
            return np.sum(data, axis = 1, keepdims = True) / np.shape(data)[1]
        mean = average(data)
        std = np.sqrt(average((data - mean) ** 2))
        return (data - mean)/std
        
    def randInit(self):
        L = self.L
        n = self.n
        for l in range(1, L+1):
            self.W[l] = np.random.rand(n[l], n[l-1]) / n[l-1]
            self.b[l] = np.zeros((n[l],1))

    def forward(self, test = False, test_x = None):

        def forward_single(Aprev, W, b, activation = "relu"):
            Z = np.dot(W, Aprev) + b
            if activation == "relu":
                A = DeepNN.relu(Z)
            elif activation == "sigmoid":
                A = DeepNN.sigmoid(Z)
            return Z, A

        if not test:
            A_current = DeepNN.reshape(self.x)
            self.A[0] = A_current
        else:
            A_current = DeepNN.reshape(test_x)

        L = self.L
        for l in range(1, L):
            W = self.W[l]
            b = self.b[l]
            self.Z[l], self.A[l] = forward_single(A_current, W, b)
            A_current = self.A[l]
        W = self.W[L]
        b = self.b[L]
        self.Z[L], self.A[L] = forward_single(A_current, W, b, activation = "sigmoid")
        return self.A[L]

    def update_cost(self):
        h = self.forward()
        self.h = h
        y = self.y
        m = self.m
        L = self.L
        reg = self.reg
        reg_cost = 0
        h[h>=0.99] = 0.99
        h[h<0.001] = 0.001
        for l in range(1, L+1):
            reg_cost = np.sum(np.sum(self.W[l] ** 2))
        self.cost = -1/m * np.sum(np.sum(y * np.log(h) + (1-y) * np.log(1-h)))
        #+ 0.5*reg/m * reg_cost

    def backward(self):
        
        def sigmoid_backward(dA, Z):
            A = DeepNN.sigmoid(Z)
            dZ = dA * A * (1-A)
            return dZ

        def relu_backward(dA, Z):
            dZ = dA
            dZ[Z<=0] = 0
            return dZ

        def linear_backward_single(dZ, A_prev, W, m):
            dW = 1/m * np.dot(dZ, A_prev.T) 
            db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
            dA_prev = np.dot(W.T, dZ)
            return dA_prev, dW, db

        def activation_backward_single(dA, storage, m, act):
            A_prev, W, Z = storage
            if act == "relu":
                dZ = relu_backward(dA, Z)
            elif act == "sigmoid":
                dZ = sigmoid_backward(dA, Z)
            
            dA_prev, dW, db = linear_backward_single(dZ, A_prev, W, m)
            return dA_prev, dW, db
        
        y = self.y
        m = self.m
        AL = self.h
        L = self.L
        dAL = -y / AL + (1-y)/(1-AL)
        storage = (self.A[L-1], self.W[L], self.Z[L])
        self.storage = storage
        self.dA[L-1], self.dW[L], self.db[L] = activation_backward_single(dAL, storage, m, act = "sigmoid")
        for l in range(L-1, 0, -1):
            storage = (self.A[l-1], self.W[l], self.Z[l])
            self.dA[l-1], self.dW[l], self.db[l] = activation_backward_single(self.dA[l], storage, m, "relu")

    def update_parameters(self):
        reg = self.reg
        m = self.m
        alpha = self.alpha
        for l in range(1, self.L + 1):
            self.W[l] -= alpha * (self.dW[l] + reg/m * self.W[l])
            self.b[l] -= alpha * self.db[l]
    
    def train(self):
        self.randInit()
        prev_cost = 1000000000000
        self.update_cost()
        i = 0
        while np.absolute(prev_cost - self.cost) > 1e-7:
            prev_cost = self.cost
            if i % 100 == 0:
                print("Cost after {:>4d} iterations is : {:.6f}".format(i,self.cost))
            self.backward()
            self.update_parameters()
            self.update_cost()
            i+=1
        self.check_accuracy()

    def check_accuracy(self):
        h = (self.h + 0.5)//1
        y = self.y
        m = self.m
        L = self.L
        nl = self.n[L]
        
        acc = 1/m * sum((sum(h == y) == nl)) * 100
        print("Training Accuracy = {:.2f}%".format(acc))

    def predict(self, x):
        AL = self.forward(test = True, test_x = x)
        AL = (AL + 0.5) // 1
        print("The prediction of input {} is : {}".format(x, AL))

    

    
np.seterr(under = "ignore")
np.seterr(over = "ignore")       



