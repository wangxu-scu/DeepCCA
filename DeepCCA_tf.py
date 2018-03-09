import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from pynverse import inversefunc
import datetime




def minibatches(inputs=None, batch_size=None, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]

def active_fn(inpt):
    """
    The activation function proposed in the paper
    :param inpt: input tensor
    :return: activated tensor
    """

    g = (lambda x: x ** 3 / 3.0 + x)
    s = inversefunc(g)
    act = tf.py_func(s, [inpt], tf.double)
    act = tf.cast(act, tf.float32)
    act.set_shape(inpt.shape)
    return act

#### DeepCCA class ####
class DeepCCA(object):
    """
    DeepCCA
    """

    def __init__(
            self, input_size1, input_size2, hidden_layer_sizes1, hidden_layer_sizes2, out_layer_sizes1,
            out_layer_sizes2, use_all_singular_values=True, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_1 = tf.placeholder(tf.float32, [None, input_size1], name="input_1")
        self.input_2 = tf.placeholder(tf.float32, [None, input_size2], name="input_2")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        #### model for view1 #######
        with tf.name_scope("model_1"):
            h1 = layers.fully_connected(self.input_1, hidden_layer_sizes1, activation_fn=tf.nn.sigmoid)
            o1 = layers.fully_connected(h1, out_layer_sizes1, activation_fn=None)

        #### model for view1 #######
        with tf.name_scope("model_1"):
            h2 = layers.fully_connected(self.input_2, hidden_layer_sizes2, activation_fn=tf.nn.sigmoid)
            o2 = layers.fully_connected(h2, out_layer_sizes2, activation_fn=None)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            r1 = 1e-4
            r2 = 1e-4

            H1 = tf.transpose(o1)
            H2 = tf.transpose(o2)


            m = tf.cast(self.batch_size, tf.float32)

            H1bar = H1 - (1.0 / m) * tf.matmul(H1, tf.ones([self.batch_size, self.batch_size]))
            H2bar = H2 - (1.0 / m) * tf.matmul(H2, tf.ones([self.batch_size, self.batch_size]))

            SigmaHat12 = (1.0 / (m - 1)) * tf.matmul(H1bar, tf.transpose(H2bar))
            SigmaHat11 = (1.0 / (m - 1)) * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(out_layer_sizes1)
            SigmaHat22 = (1.0 / (m - 1)) * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(out_layer_sizes2)

            # Calculating the root inverse of covariance matrices by using eigen decomposition
            [D1, V1] = tf.linalg.eigh(SigmaHat11)
            [D2, V2] = tf.linalg.eigh(SigmaHat22)

            SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.linalg.diag(D1 ** -0.5)), tf.transpose(V1))
            SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.linalg.diag(D2 ** -0.5)), tf.transpose(V2))

            Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
            Tval.set_shape([out_layer_sizes1, out_layer_sizes2])

            if use_all_singular_values:
                # all singular values are used to calculate the correlation
                # corr = tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(Tval), Tval)))
                s = tf.svd(Tval, compute_uv=False)
                corr = tf.reduce_sum(s)
            else:
                # just the top k singular values are used
                k = 10
                s = tf.svd(Tval, compute_uv=False)
                corr = tf.reduce_sum(s[0:k])

            self.loss = -corr

if __name__ == '__main__':

    ### Load Data  ############
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/MNIST_data", one_hot=False)

    train = mnist.train.images[0:50000]
    valid = np.concatenate([mnist.validation.images, mnist.train.images[50000:]], 0)
    test = mnist.test.images

    ### Create Session ##########
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=session_conf)


    #### Create DeepCCA object #######
    deepCCA = DeepCCA(input_size1=392, input_size2=392,
                      hidden_layer_sizes1=2038, hidden_layer_sizes2=1608,
                      out_layer_sizes1=50, out_layer_sizes2=50,
                      use_all_singular_values=True)
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0,rho=0.95)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(deepCCA.loss, aggregation_method=2)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    sess.run(tf.global_variables_initializer())

    def train_step(view1_batch, view2_batch):
        """
        A single training step
        """
        feed_dict = {
            deepCCA.input_1: view1_batch,
            deepCCA.input_2: view2_batch,
            deepCCA.batch_size: view1_batch.shape[0]
        }
        _, step, loss = sess.run(
            [train_op, global_step, deepCCA.loss],
            feed_dict)
        time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
        print("{}: step {}, loss {}".format(time_str, step, loss))

    def dev_step(view1_batch, view2_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            deepCCA.input_1: view1_batch,
            deepCCA.input_2: view2_batch,
            deepCCA.batch_size: view1_batch.shape[0]
        }
        step, loss= sess.run([global_step, deepCCA.loss], feed_dict)
        return loss


    ####### Training and Validating ##########
    num_epochs = 100
    batch_size = 800
    training_iters_one_epoch = train.shape[0] // batch_size
    val_iters_one_epoch = test.shape[0] // batch_size
    evaluate_every = 100

    for i in range(num_epochs):
        train_iter = minibatches(train, batch_size=batch_size)
        for it in range(training_iters_one_epoch):
        # for xs in train_iter:
            xs = train_iter.__next__()
            train_view1 = xs[:, 0:392]
            train_view2 = xs[:, 392:]
            train_step(train_view1, train_view2)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                total_val_loss = 0.
                val_iter = minibatches(valid, batch_size=batch_size)
                for it in range(val_iters_one_epoch):
                # for val_batch in val_iter:
                    val_batch = val_iter.__next__()
                    val_view1 = val_batch[:, 0:392]
                    val_view2 = val_batch[:, 392:]
                    val_loss_per_batch = dev_step(val_view1, val_view2)
                    total_val_loss += val_loss_per_batch
                val_loss = float(total_val_loss) / (val_iters_one_epoch)
                print("val_loss {}".format(val_loss))