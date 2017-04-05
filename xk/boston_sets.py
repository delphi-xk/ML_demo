# _*_ coding: utf-8 _*_

"""

boston_sets created by xiangkun on 2017/3/25

"""

import numpy as np
import tensorflow as tf
from sklearn import datasets


# boston = datasets.load_boston()
# X = boston.data
# y = boston.target
#
#
# print(X.shape, y.shape)

# print np.random.rand(100).reshape(50,2)
x = range(6)
tf_array = tf.reshape(x, [6,1])

# tf.squeeze flatten the matrix?
squeezed_x = tf.squeeze(tf_array)
transpose_x = tf.transpose(tf_array)

sess = tf.Session()
print sess.run(squeezed_x)
print "\n"
print sess.run(transpose_x)
