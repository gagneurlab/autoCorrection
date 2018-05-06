from __future__ import absolute_import
import tensorflow as tf
import numpy as np

#
# Constant values
# 
THETA = [0.0]

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)


class NB(object):
    def __init__(self, theta=None, scope='nbinom_loss/', scale_factor=1.0, 
                 debug=False, out_idx=None):

        # for numerical stability
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.theta = theta
        self.out_idx = out_idx

        with tf.name_scope(self.scope):
            # a variable may be given by user or it can be created here
            if theta is None:
                theta = tf.Variable(THETA, dtype=tf.float32, name='theta')

            # to keep dispersion always non-negative
            self.theta = tf.nn.softplus(theta)

            if self.out_idx is not None:
                self.out_idx = tf.cast(self.out_idx, tf.bool)

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = self.theta

            t1 = -tf.lgamma(y_true+theta)
            t2 = tf.lgamma(theta)
            t3 = tf.lgamma(y_true+1.0)
            t4 = -(theta * (tf.log(theta)))
            t5 = -(y_true * (tf.log(y_pred)))
            t6 = (theta+y_true) * tf.log(theta+y_pred)

            assert_ops = [
                    tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                    tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                    tf.verify_tensor_all_finite(t2, 't2 has inf/nans'),
                    tf.verify_tensor_all_finite(t3, 't3 has inf/nans'),
                    tf.verify_tensor_all_finite(t4, 't4 has inf/nans'),
                    tf.verify_tensor_all_finite(t5, 't5 has inf/nans'),
                    tf.verify_tensor_all_finite(t6, 't6 has inf/nans')
            ]

            if self.debug:
                tf.summary.histogram('t1', t1)
                tf.summary.histogram('t2', t2)
                tf.summary.histogram('t3', t3)
                tf.summary.histogram('t4', t4)
                tf.summary.histogram('t5', t5)
                tf.summary.histogram('t6', t6)

                with tf.control_dependencies(assert_ops):
                    final = t1 + t2 + t3 + t4 + t5 + t6

            else:
                final = t1 + t2 + t3 + t4 + t5 + t6

            if mean:
                 if self.out_idx is not None:
                    final = tf.boolean_mask(final, self.out_idx)
                    final = tf.reduce_mean(final)
            final = _nan2inf(final)
        return final

