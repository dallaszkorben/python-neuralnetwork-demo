import numpy as np
import unittest

class IListener:
    def update(self, arg):
        pass


class Neurnet:
    """
Neural Network Object
    parameters:
        input         (ndarray)    nxm numpy array. n=trainings number, m-input values in the neurons in the input layer
        expected        (ndarray)    nxm numpy array. n=training number, m-expected values in the neurons in the output layer
        hiddenlayers  (tupple)     the element represents the hidden Layers and their values the number of the Neurons in the layer
    Exception:
        if input parameter is not numpy.ndarray type
        if expected parameter is not numpy.ndarray type
        if input and expected parameter has different number of lines which represents the trainings
"""
    

    def __init__(self, input, expected, hiddenlayers):

        # checking parameters
        if( type(input).__name__ != "ndarray" ):
            raise Exception( "The parameter 'input' is not 'numpy.ndarray': ", type(input).__name__ )
        if( type(expected).__name__ != "ndarray" ):
            raise Exception( "The parameter 'expected' is not 'numpy.ndarray': ", type(expected).__name__ )
        if( input.shape[0] != expected.shape[0]):
            raise Exception( "Different numbers of training data in the 'input' and 'expected' parameters" )
        if( type(hiddenlayers).__name__ != "tuple" ):
            raise Exception( "The parameter 'hiddenlayer' is not 'tuple': ", type(hiddenlayers).__name__ )
        if( not all(hiddenlayers) ):
            raise Exception( "The hiddenlayers has empty or 0 values. It needs to be at least one element and all elements must be >0" )

        # creating instance variables
        self.trainings=input.shape[0]
        self.first_layer_nodes=input.shape[1]
        self.last_layer_nodes=expected.shape[1]

        self.inputlayer=input
        self.expected=expected
        self.hiddenlayers=hiddenlayers
        self.layers_number=2+len(hiddenlayers)

        np.random.seed(1)

        # generating Weight matrix
        self.weights=list()
        self.inlay=self.first_layer_nodes
        for i in range( len(hiddenlayers) ):
            self.weights.append( 2*np.random.random((self.inlay,hiddenlayers[i]))-1 )
            self.inlay=hiddenlayers[i]
        self.weights.append( 2*np.random.random((self.inlay, self.last_layer_nodes))-1 )

        # generating Layers
        self.layers=list()

        # set for listeners
        self.listeners=set()

    def addListener(self, listener):
        if(not issubclass(listener, IListener)):
            raise Exception("The parameter is not inherited from IListener when 'addListener()' method was called")
        self.listeners.add(listener)

    def dispatch(self, cycle, error):
        for listener in self.listeners:
            listener.update(self, cycle, error)

    def nonlin(self, x, deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    def feedForward(self):
        self.layers=list()
        self.layers.append( self.inputlayer )
        for i in range( len(self.hiddenlayers) ):
            self.layers.append( self.nonlin( self.layers[i].dot( self.weights[i]) ) )
        self.layers.append( self.nonlin( self.layers[-1].dot( self.weights[ len(self.hiddenlayers) ] ) ) )

    def backPropagation(self):
        self.reversed_errors=list()
        self.reversed_deltas=list()
        for i in range( self.layers_number-1, -1, -1 ):

            #output layer is handled differently
            if( i == self.layers_number-1 ):
                self.reversed_errors.append( self.expected-self.layers[i] )
                self.reversed_deltas.append( self.reversed_errors[-1]*self.nonlin(self.layers[i], True) )
            else:
                self.reversed_errors.append( self.reversed_deltas[-1].dot( self.weights[ i ].T ) )
                self.reversed_deltas.append( self.reversed_errors[-1]*self.nonlin(self.layers[i], True) )

        self.deltas=list(reversed(self.reversed_deltas))

        for i in range( len(self.weights)-1, -1, -1 ):
            self.weights[i] += self.layers[i].T.dot(self.deltas[i+1])

    def start(self, min_error, max_iteration=60000):
        for i in range(max_iteration):
            myNeurnet.feedForward()
            myNeurnet.backPropagation()

            if (i%1000) == 0:
                mean_error = np.mean(np.abs(self.reversed_errors[0]))
                self.dispatch(i, mean_error)
                if mean_error < min_error:
                    break



class PrintListener(IListener):
    def update(self, cycle, error):
        print( "cycle: {0}. Error: {1}".format(cycle, error ) )


if False:
    x=np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

    y=np.array([
    [0],
    [1],
    [1],
    [0],
])

    myNeurnet = Neurnet(x, y, (4,))
    myNeurnet.addListener(PrintListener)
    myNeurnet.start(0.002, 200000)

    print(myNeurnet.layers[2])