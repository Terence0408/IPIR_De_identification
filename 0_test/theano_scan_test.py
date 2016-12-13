import theano
import theano.tensor as TT
import numpy as np

# test value
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)

# define tensor variable
k = theano.shared(0)
n_sym =TT.iscalar("n_sym")

results, updates = theano.scan(lambda:{k:(k+1)},n_steps = n_sym)
accumulator = theano.function([n_sym],[],updates = updates, allow_input_downcast=True)

k.get_value()
accumulator(5)
k.get_value()