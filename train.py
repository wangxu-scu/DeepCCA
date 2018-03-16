import numpy as np
from model import *
from load import read_mnist
from linear_cca import linear_cca
from sklearn import svm
from sklearn.metrics import accuracy_score


n_epochs = 100
learning_rate = 0.01
momentum=0.99
batch_size = 800
outdim_size = 10
input_size1 = 784
input_size2 = 784
layer_sizes1 = [1024, 1024, 1024, outdim_size]
layer_sizes2 = [1024, 1024, 1024, outdim_size]
reg_par = 1e-4
use_all_singular_values = True


trainData, tuneData, testData = read_mnist()

dcca_model = DeepCCA(layer_sizes1, layer_sizes2,
                      input_size1, input_size2,
                      outdim_size,
                      reg_par, use_all_singular_values)


input_view1 = dcca_model.input_view1
input_view2 = dcca_model.input_view2
hidden_view1 = dcca_model.output_view1
hidden_view2 = dcca_model.output_view2
neg_corr = dcca_model.neg_corr


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(neg_corr,
                                                                        var_list=tf.trainable_variables())

tf.global_variables_initializer().run()


iterations = 0
for epoch in range(n_epochs):
    index = np.arange(trainData.num_examples)
    np.random.shuffle(index)
    trX1 = trainData.images1[index]
    trX2= trainData.images2[index]

    for start, end in zip(range(0, trainData.num_examples, batch_size),
            range(batch_size, trainData.num_examples, batch_size)):
        Xs1 = trX1[start:end]
        Xs2 = trX2[start:end]

        _, neg_corr_val = sess.run(
                                  [train_op, neg_corr],
                                   feed_dict={
                                       input_view1:Xs1,
                                       input_view2:Xs2
                                   })


        if iterations % 100 == 0:
            print("iteration:", iterations)
            print("neg_loss_for_train:", neg_corr_val)
            tune_neg_corr_val = sess.run(neg_corr,
                feed_dict={
                    input_view1: tuneData.images1,
                    input_view2: tuneData.images2
                })
            print("neg_loss_for_tune:", tune_neg_corr_val)

        iterations += 1


################# Linear CCA #############################

X1proj, X2proj = sess.run(
                        [hidden_view1, hidden_view2],
                        feed_dict={
                            input_view1: trainData.images1,
                            input_view2: trainData.images2
                        })
XV1proj, XV2proj = sess.run(
                        [hidden_view1, hidden_view2],
                        feed_dict={
                            input_view1: tuneData.images1,
                            input_view2: tuneData.images2
                        })
XTe1proj, XTe2proj = sess.run(
                        [hidden_view1, hidden_view2],
                        feed_dict={
                            input_view1: testData.images1,
                            input_view2: testData.images2
                        })
print("Linear CCA started!")
w = [None, None]
m = [None, None]
w[0], w[1], m[0], m[1] = linear_cca(X1proj, X2proj, 10)
print("Linear CCA ended!")
X1proj -= m[0].reshape([1, -1]).repeat(len(X1proj), axis=0)
X1proj = np.dot(X1proj, w[0])

XV1proj -= m[0].reshape([1, -1]).repeat(len(XV1proj), axis=0)
XV1proj = np.dot(XV1proj, w[0])

XTe1proj -= m[0].reshape([1, -1]).repeat(len(XTe1proj), axis=0)
XTe1proj = np.dot(XTe1proj, w[0])

trainLable = trainData.labels.astype('float')
tuneLable = tuneData.labels.astype('float')
testLable = testData.labels.astype('float')


################# SVM classify #############################

print('training SVM...')
clf = svm.LinearSVC(C=0.01, dual=False)
clf.fit(X1proj, trainLable.ravel())

p = clf.predict(XTe1proj)
test_acc = accuracy_score(testLable, p)
p = clf.predict(XV1proj)
valid_acc = accuracy_score(tuneLable, p)
print('DCCA: tune acc={}, test acc={}'.format(valid_acc, test_acc))

