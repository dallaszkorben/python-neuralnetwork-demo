import numpy as np

"""
Basic Neural Network example with 1 hidden layer.
    Input layer: 3 neurons
    Hidden layer: 4 neurons
    Output layer: 1 neuron
"""

# nonlinear function which is on the end of the Neurons
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

# input - every line represents a training data set, every training data set has 3 inputs
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

# output - every line represents an expecting output for a traning data set
y = np.array([[0],
            [1],
            [1],
            [0]])

# random seed for test purpose. It couses same random number sequence after every run
np.random.seed(1)

# before the first cycle the weights need to be initialized by random values ( mean 0)
# normal distribution with mean of 0 over the range [-1, 1)
w01 = 2*np.random.random((3,4)) - 1
w12 = 2*np.random.random((4,1)) - 1

# we use a cycle to get closer the expected value instead of checking the mean error
for j in range(60000):

    # -----------------
    #
    # Feed forward
    #
    # -----------------
    # meaning: we send the input through the Neural Network
    # -----------------------------------------------------
    #
    # Matrix lines for the training sets, columns are for the Nodes in the training set
    #
    #0. layer (input-4x1) is always the input
    l0 = X
    #1. layer (hidden-4x4) => nonlin(Σw.σ)
    l1 = nonlin(np.dot(l0,w01))
    #2. layer (output-4x1) => nonlin(Σw.σ)
    l2 = nonlin(np.dot(l1,w12))


    # -----------------
    #
    # Back Propagation
    #
    # -----------------
    # meaning: we propagate back the error from the output till the input
    # -------------------------------------------------------------------
    #
    # the error on the output layer value compared to the expected value
    l2_error = y - l2

    # delta on the output layer - calculated differently than on the other layers
    # this is the walue we should modify the w12 weight
    l2_delta = l2_error*nonlin(l2,deriv=True) # [4x1]

    # we print out the mean error after every 10000 cycle
    if (i % 10000) == 0:
        print ("Error:{0}".format( str(np.mean(np.abs(l2_error)))))

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(w12.T)

    # delta on the hidden layer
    # this is the value we should modify the w01 weight
    l1_delta = l1_error * nonlin(l1,deriv=True)

    # weight modification
    w12 += l1.T.dot(l2_delta)
    w01 += l0.T.dot(l1_delta)


# after the last cycle prints out the output
print(l2)