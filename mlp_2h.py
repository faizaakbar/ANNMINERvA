#!/usr/bin/env python
"""
A multilayer perceptron example using Theano.

Usage:
    python mlp.py [-t/--train]
                  [-p/--predict]
                  [-d/--data <path-to-dataset>]
                  [-n/--nepochs <# of epochs>]
                  [--L1 <L1 regularization parameter>]
                  [--L2 <L2 regularization parameter>]
                  [-a/--alpha <learning rate factor>]
                  [-k/--kconst <learning rate exponential decay>]

    Default train False
            predict False
            dataset: "./skim_data_target0.pkl.gz"
            N epochs: 1000
            L1: 0.000
            L2: 0.001
            alpha: 0.01
            kconst: 0.01

Note:
    * The prediction requires a stored model.

Math:
    f(x) = G(b^{(2)} + W^{(2)}(s(b^{(1)} + W^{(1)} x)))

**Much** code from:
    http://deeplearning.net/

References:
    _Pattern Recognition and Machine Learning_, Christophe Bishop, Sec. 5
"""
from __future__ import print_function

import os
import timeit
import cPickle

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data


BEST_PICKLEJAR = 'mlp_2h_best_model.pkl'
NINPUT = 2200
NOUT = 6
NHIDDEN1 = 1000
NHIDDEN2 = 500


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None, activation=T.tanh):
        """
        MLP hidden layer - units are fully connected and have a sigmoidal
        activation function. weight matrix `W` is of shape (n_in, n_out) and
        the bias vector `b` is of shape (n_out,)

        the nonlinearity used by default is `tanh`

        hidden unit activation by default is `tanh(dot(input, W) + b)`

        * rng - numpy.random.RandomState - a random number generator for weight
        initialization

        * input - theano.tensor.dmatrix - symbolic tensor of shape
        (n_examples, n_in)

        * n_in - int - (1d) dimensionality (vect length) of the input

        * n_out - int - number (vector length) of hidden units

        * activation - theano.Op or function - non-linearity to be applied to
        the hidden layer (we may also pass `None` for a linear output, or no
        non-linearity applied before passing the output along)
        """
        self.input = input

        # `W` is initalized with `W_values` which is uniformly sampled from
        # [-sqrt(6./(n_in + n_out)), sqrt(6./(n_in + n_out))] for a
        # `tanh` activation function.
        #
        # the output of unifrom is converted using `asarray` to dtype
        # theano.config.floatX so the code may be run on a GPU
        #
        # note: optimal initialization is dependent on (among other things) the
        # activation function used
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6.0 / (n_in + n_out)),
                    high=numpy.sqrt(6.0 / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """
    multi-layer perceptron with two hidden layers
    """

    def __init__(self, rng, input, n_in,
                 n_hidden1, n_hidden2, n_out,
                 W_hidden1=None, b_hidden1=None,
                 W_hidden2=None, b_hidden2=None,
                 W_logreg=None, b_logreg=None):
        """
        initialize the params of the mlp

        rng - numpy.random.RandomState - a random number generator used to
        initialize weights

        input - T.TensorType - symbolic var that describes the input of the
        architecture (one minibatch)

        n_in - int - number of input units (dimension of the space of the
        datapoints)

        n_hidden1 - int - number of hidden units in 1st hidden layer
        n_hidden2 - int - number of hidden units in 2nd hidden layer

        n_out - int - number of output units (dimension of the space of the
        labels)

        W_hidden and b_hidden - theano tensors holding trained weights and
        biases for the hidden layer. if we pass `None`, we are telling the
        MLP to initialize these values with the random number generator `rng`

        W_logreg and b_logreg - theano tensors holding trained weights and
        biases for the logreg layer. if we pass `None`, we are telling the
        MLP to initialize these values with the random number generator `rng`
        """
        # this is a two-hidden-layer MLP, so we will create a HiddenLayer
        # with `tanh` activation connected to the logistic regression layer.
        # the activation function can be replaced by a sigmoid (or something
        # else)
        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden1,
            W=W_hidden1,
            b=b_hidden1,
            activation=T.tanh
        )
        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden1,
            n_out=n_hidden2,
            W=W_hidden2,
            b=b_hidden2,
            activation=T.tanh
        )

        # the logistic regression layer gets as input the hidden units of the
        # hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden2,
            n_out=n_out,
            W=W_logreg,
            b=b_logreg
        )

        # L1 norm - one regularization option is to enforce L1 norm be small.
        # here, we just use the sum of the absolute values of the elements
        # of the Weights tensors
        self.L1 = (
            abs(self.hiddenLayer1.W).sum() +
            abs(self.hiddenLayer2.W).sum() +
            abs(self.logRegressionLayer.W).sum()
        )

        # L2 norm - one regularization option is to enforce L2 norm be small.
        # here, we use the sum of the squares of the elements of the Weights
        # tensors
        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum() +
            (self.hiddenLayer2.W ** 2).sum() +
            (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log-likelihood of the MLP is given by the negative LL of the
        # output of the model, computed in the logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        # same for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the params of the model are the params of the two layers composing it
        self.params = self.hiddenLayer1.params + \
            self.hiddenLayer2.params + \
            self.logRegressionLayer.params

        self.input = input


def train_mlp(alpha0=0.01, invk=0.01, L1_reg=0.00, L2_reg=0.0001,
              n_epochs=1000, dataset='skim_data_target0.pkl.gz',
              batch_size=20):
    """
    stochastic gradient descent optimization for a MLP using MNIST

    the learning rate will decay according to alpha0 / (1 - invk * batch)

    L1_reg - float - L1-norm's weight when added to the cost

    l2_reg - float - L2-norm's weight when added to the cost

    n_epochs - int - maximal number of epochs to run the optimizer

    dataset - string - path to the data
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]
    valid_set_x, valid_set_y = datasets[2]

    # compute number of minibatches for each stage
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print("...building the model")
    print("    alpha0 = %f, k = %f, lr = alpha0 / (1 + k * epoch)" %
          (alpha0, invk))

    # symbolic vars for the data
    index = T.lscalar()      # minibatch index
    tepoch = T.lscalar()     # the epoch as a Theano scalar
    x = T.matrix('x')        # rasterized image data
    y = T.ivector('y')       # labels 1, 2, 3, 4, 5 (plus 0)

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=NINPUT,
        n_hidden1=NHIDDEN1,
        n_hidden2=NHIDDEN2,
        n_out=NOUT
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); the cost is
    # expressed here symbolically
    cost = (
        classifier.negative_log_likelihood(y) +
        L1_reg * classifier.L1 +
        L2_reg * classifier.L2_sqr
    )

    # theano function that computes the mistakes made by the model over a
    # minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of the cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update_expression) pairs
    updates = [
        (param, param - (alpha0 / (1 + invk * tepoch)) * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compile a Theano function to return the cost and update the parameters of
    # the model based on the rules defined in `updates`
    train_model = theano.function(
        inputs=[index, tepoch],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print("...training")

    # early-stopping parameters
    patience = 10000        # min. number of examples
    patience_increase = 2   # wait this much longer when a new best is found
    improvement_threshold = 0.9999    # 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1

        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, epoch)
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                # zero-one loss on the validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print("epoch %i, minibatch %i/%i, learning rate %f, "
                      "minibatch avg cost %f, "
                      "validation err %f %%" %
                      (epoch, minibatch_index + 1, n_train_batches,
                       alpha0 / (1 + invk * epoch),
                       minibatch_avg_cost, this_validation_loss * 100.0))

                # if this is our best validation score so far...
                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < \
                            best_validation_loss * improvement_threshold:
                        print("   Current validation loss: ",
                              this_validation_loss)
                        print("   Best val loss times thresh: ",
                              best_validation_loss * improvement_threshold)
                        print("   Current patience: ", patience)
                        patience = max(patience, iter_num * patience_increase)
                        print("   New patience: ", patience)

                    best_validation_loss = this_validation_loss
                    best_iter = iter_num

                    # test on the test set
                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print("   epoch %i, minibatch %i/%i, test error of "
                          "best model %f %%" %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.0))

            if patience < iter_num:
                print("Reached early stopping condition...")
                print("Patience: ", patience)
                print("Iter Num: ", iter_num)
                done_looping = True
                break

    end_time = timeit.default_timer()
    print("Optimization complete. Best validation score of %f %% "
          "obtained at iteration %i, with test performance %f %%" %
          (best_validation_loss * 100.0, best_iter + 1, test_score * 100.0))
    print('The code for file ' +
          os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.0))

    # save the best model - can't seem to pickle the whole model here, rather
    # than debug, let's just save the params and use them to re-create the
    # model
    params = classifier.params
    with open(BEST_PICKLEJAR, 'w') as f:
        cPickle.dump(params, f)


def predict(dataset):
    """
    example of loading and running a model
    """
    # test on some examples from the test set
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    pars = cPickle.load(open(BEST_PICKLEJAR))

    # load the saved weights and bias vectors
    for i, p in enumerate(pars):
        print("Checking loaded parameter %i type and shape..." % i)
        print(type(p))
        print(p.eval().shape)

    # symbolic vars for the data
    x = T.matrix('x')        # rasterized image data
    rng = numpy.random.RandomState(1234)

    # use our loaded params to init the model
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=NINPUT,
        n_hidden1=NHIDDEN1,
        n_hidden2=NHIDDEN2,
        n_out=NOUT,
        W_hidden1=pars[0],
        b_hidden1=pars[1],
        W_hidden2=pars[2],
        b_hidden2=pars[3],
        W_logreg=pars[4],
        b_logreg=pars[5]
    )

    # compile a predictor fn
    #  use classifier.logRegressionLayer.p_y_given_x to look at the full
    #  softmax array prior to the argmax call.
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred
        # outputs=classifier.logRegressionLayer.p_y_given_x
    )
    predict_model_show_probs = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.p_y_given_x
    )

    show_size = 50

    predicted_values_probs = predict_model_show_probs(test_set_x[:show_size])
    predicted_values = predict_model(test_set_x[:show_size])
    print("Predicted values (probs) for the first %d:" % (show_size))
    print(predicted_values_probs)
    print("Predicted values for the first %d:" % (show_size))
    print(predicted_values)
    print("Actual values:")
    print(T.cast(test_set_y, 'int32').eval()[:show_size])


if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option('-d', '--data', dest='dataset',
                      default='./skim_data_target0.pkl.gz', help='Data set',
                      metavar='DATASET')
    parser.add_option('-n', '--nepochs', dest='n_epochs', default=1000,
                      help='Number of epochs', metavar='N_EPOCHS',
                      type='int')
    parser.add_option('-t', '--train', dest='do_train', default=False,
                      help='Run the training', metavar='DO_TRAIN',
                      action='store_true')
    parser.add_option('-p', '--predict', dest='do_predict', default=False,
                      help='Run a prediction', metavar='DO_PREDICT',
                      action='store_true')
    parser.add_option('-a', '--alpha', dest='alpha0', default=0.01,
                      help='Learning rate alpha', metavar='LRATEALPHA',
                      type='float')
    parser.add_option('-k', '--kconst', dest='invk', default=0.01,
                      help='Learning rate decay constant', metavar='LRATEK',
                      type='float')
    parser.add_option('--L1', dest='l1', default=0.0,
                      help='L1 regularization', metavar='L1REG',
                      type='float')
    parser.add_option('--L2', dest='l2', default=0.001,
                      help='L2 regularization', metavar='L2REG',
                      type='float')
    (options, args) = parser.parse_args()

    if not options.do_train and not options.do_predict:
        print("\nMust specify at least either train or predict:\n\n")
        print(__doc__)

    if options.do_train:
        train_mlp(alpha0=options.alpha0,
                  invk=options.invk,
                  L1_reg=options.l1,
                  L2_reg=options.l2,
                  dataset=options.dataset,
                  n_epochs=options.n_epochs)

    if options.do_predict:
        predict(dataset=options.dataset)
