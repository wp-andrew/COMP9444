"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

You do not need to import any other libraries for this assignment.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create in part II.
"""

import tensorflow as tf

""" PART I """


def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af


def add_consts_with_placeholder():
    """ 
    Construct a TensorFlow graph that constructs 2 constants, 5.1, 1.0 and one
    TensorFlow placeholder of type tf.float32 that accepts a scalar input,
    and adds these three values together, returning as a tuple, and in the
    following order:
    (the resulting tensor, the constructed placeholder).
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.placeholder(tf.float32)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af, c3


def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    c         = tf.constant(0.0)
    out_value = tf.maximum(c, in_value)
    return out_value


def my_perceptron(x):
    """
    Implement a single perception that takes four inputs and produces one output,
    using the RelU activation function you defined previously.

    Specifically, implement a function that takes a list of 4 floats x, and
    creates a tf.placeholder the same length as x. Then create a trainable TF
    variable that for the weights w. Ensure this variable is
    set to be initialized as all ones.

    Multiply and sum the weights and inputs following the peceptron outlined in the
    lecture slides. Finally, call your relu activation function.
    hint: look at tf.get_variable() and the initalizer argument.
    return the placeholder and output in that order as a tuple

    Note: The code will be tested using the following init scheme
        # graph def (your code called)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # tests here

    """
    i   = tf.placeholder(tf.float32, [x])
    w   = tf.get_variable('w', [x], tf.float32, tf.ones_initializer())
    out = my_relu(tf.reduce_sum(tf.multiply(i, w)))
    return i, out


""" PART II """
fc_count = 0  # count of fully connected layers. Do not remove.


def input_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")


def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")


def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    # initialize weights and biases
    w = tf.Variable(tf.random_normal([784, layersize], seed=13))
    b = tf.Variable(tf.random_normal([layersize], seed=13))
    
    # output layer with softmax activation
    logits = tf.add(tf.matmul(X, w), b)
    preds  = tf.nn.softmax(logits)
    
    # cross entropy cost function
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
    batch_loss     = tf.reduce_mean(batch_xentropy)
    
    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    # initialize weights and biases
    w1 = tf.Variable(tf.random_normal([784, hiddensize], seed=13))
    w2 = tf.Variable(tf.random_normal([hiddensize, outputsize], seed=13))
    b1 = tf.Variable(tf.random_normal([hiddensize], seed=13))
    b2 = tf.Variable(tf.random_normal([outputsize], seed=13))
    
    # hidden layer with ReLU activation
    hiddenlogits = tf.add(tf.matmul(X, w1), b1)
    hiddenpreds  = tf.nn.relu(hiddenlogits)
    
    # output layer with softmax activation
    logits = tf.add(tf.matmul(hiddenpreds, w2), b2)
    preds  = tf.nn.softmax(logits)
    
    # cross entropy cost function
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
    batch_loss     = tf.reduce_mean(batch_xentropy)
    
    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="SAME"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """
    # initialize shapes of convolutional layers
    conv1shape = [filter_shape[0], filter_shape[1], 1, convlayer_sizes[0]]
    conv2shape = [filter_shape[0], filter_shape[1], convlayer_sizes[0], convlayer_sizes[1]]
    
    # initialize weights and biases
    conv1w = tf.Variable(tf.truncated_normal(conv1shape, stddev=0.03))
    conv2w = tf.Variable(tf.truncated_normal(conv2shape, stddev=0.03))
    w      = tf.Variable(tf.random_normal([784 * convlayer_sizes[1], outputsize], seed=13))
    conv1b = tf.Variable(tf.truncated_normal([convlayer_sizes[0]], stddev=0.01))
    conv2b = tf.Variable(tf.truncated_normal([convlayer_sizes[1]], stddev=0.01))
    b      = tf.Variable(tf.random_normal([outputsize], seed=13))
    
    # first convolutional layer with ReLU activation
    conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(X, conv1w, strides=[1, 1, 1, 1], padding=padding), conv1b))
    
    # second convolutional layer with ReLU activation
    conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(conv1, conv2w, strides=[1, 1, 1, 1], padding=padding), conv2b))
    
    # flatten out the output from the final convolutional layer
    conv2X = tf.reshape(conv2, [-1, 784 * convlayer_sizes[1]])
    
    # output layer with softmax activation
    logits = tf.add(tf.matmul(conv2X, w), b)
    preds  = tf.nn.softmax(logits)
    
    # cross entropy cost function
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
    batch_loss     = tf.reduce_mean(batch_xentropy)
    
    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
