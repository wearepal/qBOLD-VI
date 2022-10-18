# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Contains logit Normal distribution

import tensorflow as tf
import tensorflow_probability as tfp
import math
import numpy as np
from tensorflow import keras


# A layer for performing the reparameterisation trick with a logit Normal distribution
class ReparamTrickLayer(keras.layers.Layer):
    def __init__(self, prob_dist):
        assert issubclass(type(prob_dist), LogitN)
        # Store the distributional form class
        self._prob_dist = prob_dist
        super().__init__()

    # Draw samples of OEF and DBV from the predicted distributions
    def call(self, inputs, *args, **kwargs):
        return self._prob_dist.sample(inputs)


def gaussian_nll(obs, mean, log_std):
    return -(-log_std - (1.0 / 2.0) * ((obs - mean) / tf.exp(log_std)) ** 2)


def logit(signal):
    # Inverse sigmoid function
    return tf.math.log(signal / (1.0 - signal))


class LogitN:
    def __init__(self):
        self._oef_range = 0.8
        self._min_oef = 0.04
        self._dbv_range = 0.2
        self._min_dbv = 0.001

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
        return tf.tanh(pred_offdiag) * np.exp(-2.0)

    def inv_transform_std(self, std):
        return tf.math.atanh((std + 1.0) / 3.0)

    def log_prob(self, observations, predicted_params):
        original_shape = tf.shape(predicted_params)[0:4]
        # Convert our predicted parameters
        predicted_params = tf.reshape(predicted_params, (-1, 4))
        oef_mean = predicted_params[:, 0]
        oef_log_std = self.transform_std(predicted_params[:, 1])
        dbv_mean = predicted_params[:, 2]
        dbv_log_std = self.transform_std(predicted_params[:, 3])

        # Backwards transform the true values (so we can define our distribution in the parameter space)
        x = self.backwards_transform(observations[:, 0:2], False)
        loss_oef = gaussian_nll(logit(x[:, 0]), oef_mean, oef_log_std)
        loss_dbv = gaussian_nll(logit(x[:, 1]), dbv_mean, dbv_log_std)
        loss = loss_oef + loss_dbv + tf.reduce_sum(tf.math.log(x * (1.0 - x)), -1)
        loss = tf.reshape(loss, original_shape)
        return loss

    def sample(self, inputs):
        input, mask = inputs
        oef_sample, dbv_sample = self._sample_base(input)
        samples = tf.stack([oef_sample, dbv_sample], -1)
        samples = self.forward_transform(samples)
        return samples

    def _sample_base(self, input):
        oef_sample = input[:, :, :, :, 0] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
            self.transform_std(input[:, :, :, :, 1]))
        dbv_sample = input[:, :, :, :, 2] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
            self.transform_std(input[:, :, :, :, 3]))
        return oef_sample, dbv_sample

    @property
    def num_params(self):
        # return the number of parameters for an instance of this distribution
        return 4

    def get_vars(self, y_pred):
        oef_log_std = self.transform_std(y_pred[:, 1])
        dbv_log_std = self.transform_std(y_pred[:, 3])
        oef_var = tf.exp(oef_log_std * 2.0)
        dbv_var = tf.exp(dbv_log_std * 2.0)
        return oef_var, dbv_var

    @staticmethod
    def _kl_analytic(q_mean, q_log_std, p_mean, p_log_std):
        q = tfp.distributions.LogitNormal(loc=q_mean, scale=tf.exp(q_log_std))
        p = tfp.distributions.LogitNormal(loc=p_mean, scale=tf.exp(p_log_std))
        return q.kl_divergence(p)

    def _kl_divergence(self, q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, p_oef_mean, p_oef_log_std,
                       p_dbv_mean, p_dbv_log_std, mask):
        q_oef_log_std, q_dbv_log_std = tf.split(self.transform_std(tf.concat([q_oef_log_std, q_dbv_log_std], -1)), 2,
                                                -1)
        p_oef_log_std, p_dbv_log_std = tf.split(self.transform_std(tf.concat([p_oef_log_std, p_dbv_log_std], -1)),
                                                2, -1)
        kl_oef = LogitN._kl_analytic(q_oef_mean, q_oef_log_std, p_oef_mean, p_oef_log_std)
        kl_dbv = LogitN._kl_analytic(q_dbv_mean, q_dbv_log_std, p_dbv_mean, p_dbv_log_std)

        kl_op = (kl_oef + kl_dbv)

        # Mask the KL
        kl_op = tf.where(mask > 0, kl_op, tf.zeros_like(kl_op))
        return kl_op

    def kl_divergence_pop(self, true, predicted, return_mean):
        q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, p_oef_mean, p_oef_log_std, \
        p_dbv_mean, p_dbv_log_std = tf.split(predicted, 8, -1)
        _, _, _, _, mask = tf.split(true, 5, -1)
        kl = LogitN._kl_divergence(q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, p_oef_mean, p_oef_log_std,
                                   p_dbv_mean, p_dbv_log_std, mask)

        # Not using this component
        if False:
            ig_dist = tfp.distributions.InverseGamma(1.0, 2.0)
            prior_cost = -ig_dist.log_prob(tf.exp(tf.reduce_mean(p_dbv_log_std) * 2.0))
            prior_cost = prior_cost - ig_dist.log_prob(tf.exp(tf.reduce_mean(p_oef_log_std) * 2.0))
            prior_cost = prior_cost * tf.cast(tf.shape(predicted)[0], tf.float32)

        if return_mean:
            return (tf.reduce_sum(kl)) / tf.reduce_sum(mask)
        else:
            return kl

    def kl_divergence(self, true, predicted, return_mean):
        # Calculate the kullback-leibler divergence between the posterior and prior distibutions
        q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std = tf.split(predicted, 4, -1)
        p_oef_mean, p_oef_log_std, p_dbv_mean, p_dbv_log_std, mask = tf.split(true, 5, -1)
        kl = self._kl_divergence(q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, p_oef_mean, p_oef_log_std,
                                 p_dbv_mean, p_dbv_log_std, mask)
        if return_mean:
            return (tf.reduce_sum(kl)) / tf.reduce_sum(mask)
        else:
            return kl

    def kl_divergence_mog(self, true, predicted, return_mean, mog_components):
        _, _, _, _, mask = tf.split(true, 5, -1)
        distributions = tf.split(predicted, mog_components + 1, -1)
        q_distribution = distributions[0]
        entropy = self.transform_std(q_distribution[:, :, :, :, 1]) + self.transform_std(q_distribution[:, :, :, :, 3])
        oef_sample = q_distribution[:, :, :, :, 0] + tf.random.normal(tf.shape(q_distribution[:, :, :, :, 0])) * tf.exp(
            self.transform_std(q_distribution[:, :, :, :, 1]))
        dbv_sample = q_distribution[:, :, :, :, 2] + tf.random.normal(tf.shape(q_distribution[:, :, :, :, 0])) * tf.exp(
            self.transform_std(q_distribution[:, :, :, :, 3]))

        def gaussian_nll(sample, mean, log_std):
            return -(-self.transform_std(log_std) - (1.0 / 2.0) * (
                    (sample - mean) / tf.exp(self.transform_std(log_std))) ** 2)

        kl_op = entropy * -1
        for i in range(self._mog_components):
            kl_op = kl_op + gaussian_nll(oef_sample, distributions[i + 1][:, :, :, :, 0],
                                         distributions[i + 1][:, :, :, :, 1]) / float(mog_components)
            kl_op = kl_op + gaussian_nll(dbv_sample, distributions[i + 1][:, :, :, :, 2],
                                         distributions[i + 1][:, :, :, :, 3]) / float(mog_components)
        kl = tf.expand_dims(kl_op, -1)
        if return_mean:
            return (tf.reduce_sum(kl)) / tf.reduce_sum(mask)
        else:
            return kl

    def create_samples(self, predicted_params, mask, no_samples):
        rpl = ReparamTrickLayer(self)
        samples = []
        for i in range(no_samples):
            samples.append(rpl([predicted_params, mask]))
        samples = tf.stack(samples, -1)
        return samples


class LogitMVN(LogitN):
    def __init__(self):
        super().__init__()

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

    def log_prob(self, observations, predicted_params):
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
            return -(-tf.math.log(2.0 * math.pi) - 0.5 * log_det - 0.5 * squared_whitened_residual)

        # Backwards transform the true values (so we can define our distribution in the parameter space)
        x = self.backwards_transform(observations[:, 0:2], False)
        epsilon = 1e-6
        x = tfp.math.clip_by_value_preserve_gradient(x, epsilon, 1.0 - epsilon)
        loss = gaussian_nll_chol(logit(x), tf.stack([oef_mean, dbv_mean], -1), oef_log_std,
                                 self.transform_offdiag(predicted_params[:, 4]), dbv_log_std)
        loss = loss + tf.reduce_sum(tf.math.log(x) + tf.math.log(1.0 - x), -1)
        loss = tf.reshape(loss, original_shape)
        return loss

    def _sample_base(self, input):
        z = tf.random.normal(tf.shape(input[:, :, :, :, :2]))
        oef_sample = input[:, :, :, :, 0] + z[:, :, :, :, 0] * tf.exp(
            self.transform_std(input[:, :, :, :, 1]))
        # Use the DBV mean and draw sample correlated with the oef one (via cholesky of cov term)
        dbv_sample = input[:, :, :, :, 2] + \
                     z[:, :, :, :, 0] * self.transform_offdiag(input[:, :, :, :, 4]) + \
                     z[:, :, :, :, 1] * tf.exp(self.transform_std(input[:, :, :, :, 3]))
        return oef_sample, dbv_sample

    @property
    def num_params(self):
        # return the number of parameters for an instance of this distribution
        return 5

    def get_vars(self, y_pred):
        oef_log_std = self.transform_std(y_pred[:, 1])
        dbv_log_std = self.transform_std(y_pred[:, 3])
        oef_var = tf.exp(oef_log_std) ** 2
        dbv_var = tf.exp(dbv_log_std) ** 2 + y_pred[:, 4] ** 2
        return oef_var, dbv_var

    def kl_divergence_pop(self, true, predicted):
        raise NotImplementedError()

    def kl_divergence(self, true, predicted, return_mean, no_samples=70):
        kl_op = self.mvg_kl_samples(true, predicted, no_samples=no_samples)
        _, mask = tf.split(true, [5, 1], -1)
        kl_op = tf.where(mask > 0, kl_op, tf.zeros_like(kl_op))
        if return_mean:
            return tf.reduce_sum(kl_op) / tf.reduce_sum(mask)
        else:
            return kl_op

    def mvg_kl_samples(self, prior, pred, no_samples=50):
        prior_dist, mask = tf.split(prior, [5, 1], -1)
        samples = self.create_samples(pred, mask, no_samples)

        log_q = [-self.log_prob(tf.reshape(samples[:, :, :, :, :, x], (-1, 2)), tf.stop_gradient(pred)) for x in
                 range(no_samples)]
        log_p = [-self.log_prob(tf.reshape(samples[:, :, :, :, :, x], (-1, 2)), prior_dist) for x in range(no_samples)]

        log_q = tf.stack(log_q, -1)
        log_p = tf.stack(log_p, -1)

        # finite_mask = tf.logical_and(tf.math.is_finite(log_q), tf.math.is_finite(log_p))
        kl_op = log_q - log_p
        """
        tf.keras.backend.print_tensor(tf.reduce_mean(kl_op))
        kl_op = tf.where(tf.math.is_finite(kl_op), kl_op, tf.zeros_like(kl_op))
        kl_op = tf.reduce_sum(kl_op, -1, keepdims=True) / tf.reduce_sum(tf.cast(tf.math.is_finite(kl_op), tf.float32), -1, keepdims=True)
        """
        kl_op = tf.reduce_mean(kl_op, axis=-1, keepdims=True)
        return kl_op

    def _kl_analytic(self, prior, pred):
        # This implementation ignores the covariance between OEF and DBV in the KL
        raise NotImplementedError()

        q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, q_oef_dbv_cov = tf.split(pred, 5, -1)
        p_oef_mean, p_oef_log_std, p_dbv_mean, p_dbv_log_std, p_oef_dbv_cov, mask = tf.split(prior, 6, -1)

        q_oef_log_std, q_dbv_log_std = tf.split(self.transform_std(tf.concat([q_oef_log_std, q_dbv_log_std], -1)), 2,
                                                -1)
        p_oef_log_std, p_dbv_log_std = tf.split(self.transform_std(tf.concat([p_oef_log_std, p_dbv_log_std], -1)), 2,
                                                -1)
        q_oef_dbv_cov = self.transform_offdiag(q_oef_dbv_cov)
        p_oef_dbv_cov = self.transform_offdiag(p_oef_dbv_cov)

        q_dbv_std = tf.sqrt(tf.exp(q_dbv_log_std) ** 2 + q_oef_dbv_cov ** 2)
        q_oef_std = tf.exp(q_oef_log_std)
        p_dbv_std = tf.sqrt(tf.exp(p_dbv_log_std) ** 2 + p_oef_dbv_cov ** 2)
        p_oef_std = tf.exp(p_oef_log_std)
        q_oef = tfp.distributions.LogitNormal(loc=q_oef_mean, scale=q_oef_std)
        p_oef = tfp.distributions.LogitNormal(loc=p_oef_mean, scale=p_oef_std)
        kl_oef = q_oef.kl_divergence(p_oef)

        q_dbv = tfp.distributions.LogitNormal(loc=q_dbv_mean, scale=q_dbv_std)
        p_dbv = tfp.distributions.LogitNormal(loc=p_dbv_mean, scale=p_dbv_std)
        kl_dbv = q_dbv.kl_divergence(p_dbv)
        return kl_oef + kl_dbv

    def mvg_kl(self, true, predicted):
        raise NotImplementedError()
        # Calculate the KL in terms of the logits

        if self._use_population_prior:

            q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, q_oef_dbv_cov, p_oef_mean, p_oef_log_std, \
            p_dbv_mean, p_dbv_log_std, p_oef_dbv_cov = tf.split(predicted, 10, -1)
            _, _, _, _, _, mask = tf.split(true, 6, -1)
        else:
            q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, q_oef_dbv_cov = tf.split(predicted, 5, -1)
            p_oef_mean, p_oef_log_std, p_dbv_mean, p_dbv_log_std, p_oef_dbv_cov, mask = tf.split(true, 6, -1)

        q_oef_dbv_cov = self.transform_offdiag(q_oef_dbv_cov)
        p_oef_dbv_cov = self.transform_offdiag(p_oef_dbv_cov)
        q_oef_log_std, q_dbv_log_std = tf.split(self.transform_std(tf.concat([q_oef_log_std, q_dbv_log_std], -1)), 2,
                                                -1)
        p_oef_log_std, p_dbv_log_std = tf.split(self.transform_std(tf.concat([p_oef_log_std, p_dbv_log_std], -1)),
                                                2, -1)
        det_q = EncoderTrainer.calculate_log_chol_det(q_oef_log_std, q_dbv_log_std)
        det_p = EncoderTrainer.calculate_log_chol_det(p_oef_log_std, p_dbv_log_std)
        p_mu = tf.concat([p_oef_mean, p_dbv_mean], -1)
        q_mu = tf.concat([q_oef_mean, q_dbv_mean], -1)

        squared_residual = EncoderTrainer.squared_whitened_residual(p_mu, q_mu, p_oef_log_std, p_dbv_log_std,
                                                                    p_oef_dbv_cov)
        det_term = det_p - det_q

        inv_p_tl = 1.0 / tf.exp(p_oef_log_std)
        inv_p_br = 1.0 / tf.exp(p_dbv_log_std)
        inv_p_od = inv_p_tl * p_oef_dbv_cov * inv_p_br * -1.0
        inv_pcov_tl = inv_p_tl ** 2
        inv_pcov_br = inv_p_od ** 2 + inv_p_br ** 2
        inv_pcov_od = inv_p_tl * inv_p_od

        q_tl = tf.exp(q_oef_log_std) ** 2
        q_br = tf.exp(q_dbv_log_std) ** 2 + q_oef_dbv_cov ** 2
        q_od = q_oef_dbv_cov * tf.exp(q_oef_log_std)

        trace = inv_pcov_tl * q_tl + inv_pcov_od * q_od + inv_pcov_od * q_od + q_br * inv_pcov_br
        squared_residual = tf.expand_dims(squared_residual, -1)

        kl_op = 0.5 * (trace + squared_residual + det_term - 2.0)
        return kl_op
