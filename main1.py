import numpy as np  # import numpy library
import matplotlib.pyplot as plt # to plot error during training

class LinearLayer:

    def __init__(self, input_shape, n_out, ini_type="plain"):

        self.m = input_shape[1] 
        self.params = self.initialize_parameters(input_shape[0], n_out, ini_type)  # initialize weights and bias
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))  # create space for resultant Z output

    def forward(self, A_prev):

        self.A_prev = A_prev  # store the Activations/Training Data coming in
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']  # compute the linear function

    def backward(self, upstream_grad):
        # derivative of Cost w.r.t W
        self.dW = np.dot(upstream_grad, self.A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)

        # derivative of Cost w.r.t A_prev
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learningRate=0.1):
        self.params['W'] = self.params['W'] - learningRate * self.dW  # update weights
        self.params['b'] = self.params['b'] - learningRate * self.db  # update bias(es)

    def initialize_parameters(self, n_in, n_out, ini_type='plain'):
        params = dict()  # initialize empty dictionary of neural net parameters W and b

        if ini_type == 'plain':
            params['W'] = np.random.randn(n_out, n_in) *0.01  # set weights 'W' to small random gaussian
        elif ini_type == 'xavier':
            params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
        elif ini_type == 'he':
            params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)  # set variance of W to 2/n

        params['b'] = np.zeros((n_out, 1))    # set bias 'b' to zeros

        return params

class SigmoidLayer:

    def __init__(self, shape):
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        self.dZ = upstream_grad * self.A*(1-self.A)

def compute_cost(Y, Y_hat):
    m = Y.shape[1]

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar

    dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

    return cost, dY_hat

def think(inputs):
    compareTemp = 17 - inputs[1, 25]

    output = np.array([[1 / (1 + np.exp(-compareTemp))]])

    firstEpoch = np.array([[inputs[0]]])

    Z1.forward(firstEpoch)
    A1.forward(Z1.Z)
    
    Z2.forward(A1.A)
    A2.forward(Z2.Z)

    prediction = A2.A

    dA2 = compute_cost(Y=output, Y_hat=prediction)

    A2.backward(dA2)
    Z2.backward(A2.dZ)
    
    A1.backward(Z2.dA_prev)
    Z1.backward(A1.dZ)

    Z2.update_params(learningRate=learningRate)
    Z1.update_params(learningRate=learningRate)

    return prediction

# define training constants
learningRate = 2
numberOfEpochs = 5000
epoch_list = []

# define input 
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

X_train = X.T
Y_train = Y.T

np.random.seed(48) # set seed value so that the results are reproduceable
                  # (weights will now be initailzaed to the same pseudo-random numbers, each time)

# Layer declaration

Z1 = LinearLayer(input_shape=X_train.shape, n_out=3, ini_type="xavier")
A1 = SigmoidLayer(Z1.Z.shape)
Z2 = LinearLayer(input_shape=A1.A.shape, n_out=1, ini_type="xavier")
A2 = SigmoidLayer(Z1.Z.shape)

print(Z1.params)
print(Z2.params)

costs = [] # initially empty list, this will store all the costs after a certian number of epochs

# Start training
for epoch in range(numberOfEpochs):
    
    # ------------------------- forward-prop -------------------------
    Z1.forward(X_train)
    A1.forward(Z1.Z)
    
    Z2.forward(A1.A)
    A2.forward(Z2.Z)
    
    # ---------------------- Compute Cost ----------------------------
    cost, dA2 = compute_cost(Y=Y_train, Y_hat=A2.A)
    
    # print and store Costs every 100 iterations, add to epoch list
    if (epoch % 100) == 0:
        #print("Cost at epoch#" + str(epoch) + ": " + str(cost))
        print("Cost at epoch #{}: {}".format(epoch, cost))
        costs.append(cost)
        epoch_list.append(epoch)
    
    # ------------------------- back-prop ----------------------------
    A2.backward(dA2)
    Z2.backward(A2.dZ)
    
    A1.backward(Z2.dA_prev)
    Z1.backward(A1.dZ)
    
    # ----------------------- Update weights and bias ----------------
    Z2.update_params(learningRate=learningRate)
    Z1.update_params(learningRate=learningRate)

print(Z1.params)
print(Z2.params)

# plot the error every 100 epochs
plt.figure(figsize=(15,5))
plt.plot(epoch_list, costs)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

inputs = np.array([
    [],
    []
])

prediction = think(inputs)
