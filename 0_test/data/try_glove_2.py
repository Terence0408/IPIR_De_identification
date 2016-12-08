def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

    # Collect cooccurrences internally as a sparse matrix for
    # passable indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

# -- continued --
    for i, line in enumerate(corpus):
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

# -- continued --
        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

# -- continued --
            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend
                # we are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

def train_glove(vocab, cooccurrences, vector_size=100,
                iterations=25, **kwargs):
    # -- continued --
    vocab_size = len(vocab)

    # Word vector matrix. This matrix is (2V) * d, where N is the
    # size of the corpus vocabulary and d is the dimensionality of
    # the word vectors. All elements are initialized randomly in the
    # range (-0.5, 0.5]. We build two word vectors for each word:
    # one for the word as the main (center) word and one for the
    # word as a context word.
    #
    # It is up to the client to decide what to do with the resulting
    # two vectors. Pennington et al. (2014) suggest adding or
    # averaging the two for each word, or discarding the context
    # vectors.
    W = ((np.random.rand(vocab_size * 2, vector_size) - 0.5)
         / float(vector_size + 1))

    # Bias terms, each associated with a single vector. An array of
    # size $2V$, initialized randomly in the range (-0.5, 0.5].
    biases = ((np.random.rand(vocab_size * 2) - 0.5)
              / float(vector_size + 1))

# -- continued --
    # Training is done via adaptive gradient descent (AdaGrad). To
    # make this work we need to store the sum of squares of all
    # previous gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2,
                                      dtype=np.float64)

# -- continued --
    for i in range(iterations):
        cost = run_iter(vocab, data, **kwargs)

# -- continued --
    global_cost = 0

    # Iterate over data in random order
    shuffle(data)

# -- continued --
    for (v_main, v_context, b_main, b_context, gradsq_W_main,
         gradsq_W_context, gradsq_b_main, gradsq_b_context,
         cooccurrence) in data:

        # Calculate weight function $f(X_{ij})$
        weight = ((cooccurrence / x_max) ** alpha
                  if cooccurrence < x_max else 1)

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += cost

# -- continued --
        # Compute gradients for word vector terms.
        #
        # NB: `v_main` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for v_context, biases, etc.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

# -- continued --
        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main
                   / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context
                      / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main
                   / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context
                      / np.sqrt(gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

# -- continued --
    return global_cost

