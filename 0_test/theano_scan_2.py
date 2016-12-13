import numpy
import theano
import theano.tensor as T
from theano import config
from theano import OrderedUpdates
rng = numpy.random


#Randomly create 400 pts in a 784 Dim space and randomly
#assign to two classes
N = 400
feats = 784
D = (rng.randn(N, feats).astype(theano.config.floatX),
     rng.randint(size=N,low=0, high=2).astype(theano.config.floatX))

#Create initial learning parameters
w_value = rng.randn(feats).astype(config.floatX)
b_value = numpy.asarray(0.).astype(config.floatX)


# Declare Theano symbolic variables
x = T.fmatrix("x")
y = T.fvector("y")
w = theano.shared(w_value, name="w")
b = theano.shared(b_value, name="b")

# Training function
def oneStep(x1,y1):
   # Make graph - (?)using shared variables(?)
   p_1  = 1 / (1 + T.exp(-T.dot(x1, w)-b)) # Probability of having a one
   xent = -y1*T.log(p_1) - (1-y1)*T.log(1-p_1) # Cross-entropy
   cost = xent.mean() + 0.01*(w**2).sum() # The cost to optimize
   gw,gb = T.grad(cost, [w,b]) #Gradients
   return OrderedUpdates([(w, w - 0.1 * gw), (b, b - 0.1 * gb)])


#  Create scan/looping function for logistic regression training
training_steps = 10000
(_, batchUpdates) = theano.scan(fn=oneStep,
                                  outputs_info=[],
                                  sequences = [],
                                  non_sequences = [x,y],
                                  n_steps = training_steps)

(_, stochUpdates) = theano.scan(fn=oneStep,
                                  outputs_info=[],
                                  sequences = [x,y],
                                  non_sequences = [])

# Possible train functions
batchTrain = theano.function(inputs=[x,y], outputs=[], updates = batchUpdates)
stochTrain = theano.function(inputs=[x,y], outputs=[], updates = stochUpdates)