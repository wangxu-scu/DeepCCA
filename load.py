import scipy.io as sio
import tensorflow as tf
import numpy as np

class DataSet(object):
    
    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=tf.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                        labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                        labels.shape))
            self._num_examples = images1.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 1
            #images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            if dtype == tf.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                print("type conversion view 1")
                images1 = images1.astype(np.float32)
            
            if dtype == tf.float32 and images2.dtype != np.float32:
                print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property
    def images1(self):
        return self._images1
    
    @property
    def images2(self):
        return self._images2
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        
        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]

def read_mnist():

    data=sio.loadmat('MNIST.mat')
    
    train=DataSet(data['X1'],data['X2'],data['trainLabel'])
    
    tune=DataSet(data['XV1'],data['XV2'],data['tuneLabel'])
    
    test=DataSet(data['XTe1'],data['XTe2'],data['testLabel'])
    
    return train, tune, test


def read_xrmb():

    data=sio.loadmat('/share/data/speech-multiview/wwang5/cca/XRMBf2KALDI_window7_single.mat')
    
    train=DataSet(data['X1'],data['X2'],data['trainLabel'])
    
    tune=DataSet(data['XV1'],data['XV2'],data['tuneLabel'])
    
    test=DataSet(data['XTe1'],data['XTe2'],data['testLabel'])
    
    return train, tune, test

    
def read_flicker():

    data=sio.loadmat('/share/data/speech-multiview/wwang5/cca/VCCA/flicker/flicker_tensorflow_split1.mat')
    X1=data['X1']
    X2=data['X2']
    XV1=data['XV1']
    XV2=data['XV2']
    XTe1=data['XTe1']
    XTe2=data['XTe2']
    
    for i in range(2,11):
        
        data=sio.loadmat('/share/data/speech-multiview/wwang5/cca/VCCA/flicker/flicker_tensorflow_split' + str(i) + '.mat')
        
        X1=np.concatenate([X1, data['X1']])
        X2=np.concatenate([X2, data['X2']])
        XV1=np.concatenate([XV1, data['XV1']])
        XV2=np.concatenate([XV2, data['XV2']])
        XTe1=np.concatenate([XTe1, data['XTe1']])
        XTe2=np.concatenate([XTe2, data['XTe2']])
    
    train=DataSet(X1, X2, np.zeros(len(X1)))
    
    tune=DataSet(XV1, XV2, np.zeros(len(XV1)))
    
    test=DataSet(XTe1, XTe2, np.zeros(len(XTe1)))
    
    return train, tune, test

    


