import re
import tensorflow as tf

BATCH_SIZE = 200
MAX_WORDS_IN_REVIEW = 200  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_CLASSES = 2
NUM_LAYERS = 1
NUM_UNITS = 100

# added https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
stop_words = set({'amn', 'haven', 'both', 'having', 'such', 'their', 'own', 'over', 'go',
                  'wouldn', 'through', 'hasn', 'under', 'isn', 'below', 'something',
                  'against', 'let', 'daren', 'an', 'by', 'our', 'may', 'there', 'daresn',
                  'only', 'yesn', 'with', 'can', 'in', 'don', 'what', 'when', 'on',
                  'yourselves', 'must', 'a', 'dasn', 'or', 'doesn', 'won', 'were', 'most',
                  'if', 'does', 'shan', 'would', 'ourselves', 'all', 'might', 'ever',
                  'again', 'your', 'after', 'ma', 'hers', 'her', 'theirs', 'as', 'need',
                  'ain', 'about', 'ol', 'clock', 'gotta', 'y', 'the', 'somebody', 'they',
                  'could', 'hadn', 'fixing', 'same', 'up', 'dare', 'yes', 'needn', 'how',
                  'those', 'each', 'he', 'once', 'more', 'old', 'just', 'me', 'should',
                  'is', 'not', 'because', 'during', 'i', 'here', 'aren', 'yours', 'of',
                  't', 'its', 'herself', 'than', 'cain', 'them', 'ought', 'has', 'never',
                  'we', 'myself', 'for', 'yourself', 'now', 'which', 'while', 'shall',
                  'er', 'his', 'gimme', 'to', 'above', 'e', 'was', 'couldn', 'other',
                  'dared', 'further', 'been', 'mayn', 'gonna', 'some', 'few', 'going',
                  'someone', 'll', 're', 'weren', 'shalln', 'into', 'too', 'doing', 'had',
                  'rarely', 'everyone', 'whom', 'my', 'shouldn', 'before', 'mightn',
                  'that', 'and', 'ne', 'themselves', 'then', 'him', 'are', 'so', 'cannot',
                  'where', 've', 'do', 'o', 'very', 'have', 'himself', 'down', 'gon',
                  'she', 'ours', 'got', 'no', 'twas', 'I', 'being', 'these', 'this', 'be',
                  'finna', 'will', 'm', 'who', 'why', 'wasn', 'd', 'from', 'out', 'madam',
                  'tis', 's', 'did', 'didn', 'give', 'am', 'you', 'it', 'us', 'any',
                  'between', 'oughtn', 'off', 'mustn', 'at', 'itself', 'whomst'})
re_stop_words = re.compile(r'\b(' + '|'.join(stop_words) + ')\\W', re.I)

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # change to lower case
    processed_review = review.lower()
    # remove punctuations
    processed_review = re.sub(r'[^\w\s]', ' ', processed_review)
    # remove stop words (and contractions)
    processed_review = re_stop_words.sub('', processed_review)
    # split into list of words (and remove empty strings)
    processed_review = list(filter(None, processed_review.split(' ')))
    return processed_review

def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    # placeholders
    input_data        = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")
    labels            = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_CLASSES], name="labels")
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    # variables
    weights = tf.Variable(tf.random_normal([NUM_UNITS, NUM_CLASSES], seed=13))
    biases  = tf.Variable(tf.random_normal([NUM_CLASSES], seed=13))
    #biases  = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
    
    # 1-layer GRU network with NUM_UNITS units
    cells = []
    for _ in range(NUM_LAYERS):
        #cell = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS)
        cell = tf.contrib.rnn.GRUCell(NUM_UNITS)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    # output
    output, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    output    = tf.reduce_mean(output, axis=1)

    # accuracy
    logits        = tf.matmul(output, weights) + biases
    preds         = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    Accuracy      = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name="accuracy")

    # loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), name="loss")

    # optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
