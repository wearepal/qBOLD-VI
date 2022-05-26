# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Contains logit Normal distribution

import tensorflow as tf
import tensorflow_probability as tfp
import math
import numpy as np

def logit(signal):
    # Inverse sigmoid function
    return tf.math.log(signal / (1.0 - signal))

class LogitMVN:
    def __init__(self):
        self._oef_range = 0.8
        self._min_oef = 0.04
        self._dbv_range = 0.2
        self._min_dbv = 0.001

    @staticmethod
    def squared_whitened_residual(obs, mean, oef_log_std, dbv_log_std, oef_dbv_cov):
        out_shape = tf.shape(mean)[:-1]
        obs = tf.reshape(obs, (-1, 2))
        mean = tf.reshape(mean, (-1, 2))
        oef_log_std = tf.reshape(oef_log_std, (-1, 1))
        dbv_log_std = tf.reshape(dbv_log_std, (-1, 1))
        oef_dbv_cov = tf.reshape(oef_dbv_cov, (-1, 1))

        inv_tl = tf.exp(oef_log_std * -1.0)
        inv_br = tf.exp(dbv_log_std * -1.0)
        inv_bl = tf.exp(oef_log_std * -1.0 + dbv_log_std * -1.0) * oef_dbv_cov * -1.0
        residual = obs - mean
        whitened_residual_oef = residual[:, 0:1] * inv_tl
        whitened_residual_dbv = residual[:, 1:2] * inv_br + residual[:, 0:1] * inv_bl
        whitened_residual = tf.concat([whitened_residual_oef, whitened_residual_dbv], -1)
        squared_whitened_residual = tf.reduce_sum(tf.square(whitened_residual), -1)
        squared_whitened_residual = tf.reshape(squared_whitened_residual, out_shape)
        return squared_whitened_residual

    @staticmethod
    def calculate_log_chol_det(oef_log_std, dbv_log_std):
        # Get the log (2.0*sum log diags) of the determinant (product of squared diagonals)
        det = 2.0 * (oef_log_std + dbv_log_std)
        return det

    def logit_gaussian_mvg_log_prob(self, observations, predicted_params):
        original_shape = tf.shape(predicted_params)[0:4]
        # Convert our predicted parameters
        predicted_params = tf.reshape(predicted_params, (-1, 5))
        oef_mean = predicted_params[:, 0]
        oef_log_std = self.transform_std(predicted_params[:, 1])
        dbv_mean = predicted_params[:, 2]
        dbv_log_std = self.transform_std(predicted_params[:, 3])

        def gaussian_nll_chol(obs, mean, oef_log_std, oef_dbv_cov, dbv_log_std):
            log_det = LogitMVN.calculate_log_chol_det(oef_log_std, dbv_log_std)
            # Calculate the inverse cholesky matrix
            squared_whitened_residual = LogitMVN.squared_whitened_residual(obs, mean, oef_log_std,
                                                                                 dbv_log_std, oef_dbv_cov)
            return -(-tf.math.log(2.0*math.pi)-0.5 * log_det - 0.5 * squared_whitened_residual)

        # Scale our observation to 0-1 range
        x = self.backwards_transform(observations[:, 0:2], False)
        epsilon = 1e-6
        x = tfp.math.clip_by_value_preserve_gradient(x, epsilon, 1.0 - epsilon)
        loss = gaussian_nll_chol(logit(x), tf.stack([oef_mean, dbv_mean], -1), oef_log_std,
                                 self.transform_offdiag(predicted_params[:, 4]), dbv_log_std)
        loss = loss + tf.reduce_sum(tf.math.log(x) + tf.math.log(1.0-x), -1)
        loss = tf.reshape(loss, original_shape)
        return loss

    def forward_transform(self, logits):
        # Define the forward transform of the predicted parameters to OEF/DBV
        oef, dbv = tf.split(logits, 2, -1)
        oef = (tf.nn.sigmoid(oef) * self._oef_range) + self._min_oef
        dbv = (tf.nn.sigmoid(dbv) * self._dbv_range) + self._min_dbv
        output = tf.concat([oef, dbv], axis=-1)
        return output

    def backwards_transform(self, signal, include_logit):
        # Define how to backwards transform OEF/DBV to the same parameterisation used by the NN
        oef, dbv = tf.split(signal, 2, -1)
        oef = (oef - self._min_oef) / self._oef_range
        dbv = (dbv - self._min_dbv) / self._dbv_range
        if include_logit:
            oef = logit(oef)
            dbv = logit(dbv)
        output = tf.concat([oef, dbv], axis=-1)
        return output

    def transform_std(self, pred_stds):
        # Transform the predicted std-dev to the correct range
        return (tf.tanh(pred_stds) * 3.0) - 1.0

    def transform_offdiag(self, pred_offdiag):
        # Limit the magnitude of off-diagonal terms by pushing through a tanh
        return tf.tanh(pred_offdiag)*np.exp(-2.0)

    def inv_transform_std(self, std):
        return tf.math.atanh((std+1.0) / 3.0)

