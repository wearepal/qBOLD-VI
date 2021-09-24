# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Store the model code

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import math
import numpy as np

def logit(signal):
    # Inverse sigmoid function
    return tf.math.log(signal / (1.0 - signal))


class ReparamTrickLayer(keras.layers.Layer):
    def __init__(self, encoder_trainer):
        self._encoder_trainer = encoder_trainer
        super().__init__()

    # Draw samples of OEF and DBV from the predicted distributions
    def call(self, inputs, *args, **kwargs):
        input, mask = inputs

        if self._encoder_trainer._use_mvg:
            z = tf.random.normal(tf.shape(input[:, :, :, :, :2]))
            oef_sample = input[:, :, :, :, 0] + z[:, :, :, :, 0] * tf.exp(
                self._encoder_trainer.transform_std(input[:, :, :, :, 1]))
            # Use the DBV mean and draw sample correlated with the oef one (via cholesky of cov term)
            dbv_sample = input[:, :, :, :, 2] + \
                         z[:, :, :, :, 0] * self._encoder_trainer.transform_offdiag(input[:, :, :, :, 4]) + \
                         z[:, :, :, :, 1] * tf.exp(self._encoder_trainer.transform_std(input[:, :, :, :, 3]))

        else:
            oef_sample = input[:, :, :, :, 0] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
                self._encoder_trainer.transform_std(input[:, :, :, :, 1]))
            dbv_sample = input[:, :, :, :, 2] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
                self._encoder_trainer.transform_std(input[:, :, :, :, 3]))

            """hct_flatten = tf.keras.layers.Flatten()(input[:, :, :, :, 4:5])
            mask_flatten = tf.keras.layers.Flatten()(mask)
    
            hct_mean = tf.reduce_sum(hct_flatten * mask_flatten, -1) / (tf.reduce_sum(mask_flatten, -1) + 1e-5)
            hct_std = tf.math.reduce_std(hct_flatten, -1)
            hct_sample = tf.ones_like(oef_sample) * tf.reshape(hct_mean, (-1, 1, 1, 1))"""


        samples = tf.stack([oef_sample, dbv_sample], -1)
        # Forward transform
        samples = self._encoder_trainer.forward_transform(samples)
        return samples


class EncoderTrainer:
    def __init__(self,
                 system_params,
                 no_intermediate_layers=1,
                 no_units=10,
                 use_layer_norm=False,
                 dropout_rate=0.0,
                 activation_type='gelu',
                 student_t_df=None,
                 initial_im_sigma=0.08,
                 multi_image_normalisation=True,
                 channelwise_gating=False,
                 infer_inv_gamma=False,
                 use_mvg=True,
                 use_population_prior=True,
                 mog_components=1,
                 no_samples=1,
                 heteroscedastic_noise=True,
                 predict_log_data=True
                 ):
        self._no_intermediate_layers = no_intermediate_layers
        self._no_units = no_units
        self._use_layer_norm = use_layer_norm
        self._dropout_rate = dropout_rate
        self._activation_type = activation_type
        self._student_t_df = student_t_df
        self._initial_im_sigma = initial_im_sigma
        self._multi_image_normalisation = multi_image_normalisation
        self._system_params = system_params
        self._channelwise_gating = channelwise_gating
        self._infer_inv_gamma = infer_inv_gamma
        self._use_mvg = use_mvg
        self._use_population_prior = use_population_prior
        self._mog_components = mog_components
        self._no_samples = no_samples
        self._oef_range = 0.8
        self._min_oef = 0.04
        self._dbv_range = 0.3
        self._min_dbv = 0.002
        self._heteroscedastic_noise = heteroscedastic_noise
        self._predict_log_data = predict_log_data

    def create_encoder(self, use_conv=True, system_constants=None, gate_offset=0.0, resid_init_std=1e-1):
        """
        @param: use_conv (Boolean) : whether to use a convolution (1x1x1) or MLP model
        @params: system_constants (array): If not None, perform a dense transformation and multiply with first level representation
        @params: no_units (unsigned int): The number of units for each level of the network
        @params: use_layer_norm (Boolean) : Perform layer normalisation
        @params: dropout_rate (float, 0-1) : perform dropout
        @params: no_intermediate layers (unsigned int) : the number of extra layers apart from the first and last
        """
        l2_weight = 0.0#1e-3
        ki = tf.keras.initializers.HeNormal()
        ki_resid = tf.keras.initializers.RandomNormal(stddev=resid_init_std)

        def create_layer(_no_units, activation=self._activation_type):
            if use_conv:
                return keras.layers.Conv3D(_no_units, kernel_size=(1, 1, 1), kernel_initializer=ki,
                                           activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_weight))
            else:
                return keras.layers.Dense(_no_units, activation=activation)

        def normalise_data(_data):
            # Do the normalisation as part of the model rather than as pre-processing
            orig_shape = tf.shape(_data)
            _data = tf.reshape(_data, (-1, 11))
            _data = tf.clip_by_value(_data, 1e-2, 1e8)
            if self._multi_image_normalisation:
                # Normalise based on the mean of tau =0 and adjacent tau values to minimise the effects of noise
                _data = _data / tf.reduce_mean(_data[:, 1:4], -1, keepdims=True)
            else:
                _data = _data / tf.reduce_mean(_data[:, 2:3], -1, keepdims=True)
            # Take the logarithm
            log_data = tf.math.log(_data)
            log_data = tf.reshape(log_data, orig_shape)

            #_data = tf.reshape(_data, orig_shape)
            #return tf.concat([_data, log_data], -1)
            return log_data

        if use_conv:
            input = keras.layers.Input(shape=(None, None, None, 11), ragged=False)
        else:
            input = keras.layers.Input(shape=(11,))

        net = input

        net = keras.layers.Lambda(normalise_data)(net)

        def add_normalizer(_net):
            # Possibly add dropout and a normalization layer, depending on the dropout_rate and use_layer_norm values
            import tensorflow_addons as tfa
            if self._dropout_rate > 0.0:
                _net = keras.layers.Dropout(self._dropout_rate)(_net)
            if self._use_layer_norm:
                _net = tfa.layers.GroupNormalization(groups=1, axis=-1)(_net)
            return _net

        def create_block(_net_in, _net2_in, no_units):
            # Straightforward 1x1x1 convs for the pre-training network
            conv_layer = create_layer(no_units)
            _net = conv_layer(_net_in)

            # Apply the same 1x1x1 conv as for stream 1 for the skip connection
            _net2_skip = conv_layer(_net2_in)
            # Do a residual block
            _net2 = add_normalizer(_net2_in)
            _net2 = tf.keras.layers.Activation(self._activation_type)(_net2)
            _net2 = keras.layers.Conv3D(no_units, kernel_size=(3, 3, 1), padding='same', kernel_initializer=ki_resid, kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(
                _net2)
            _net2 = add_normalizer(_net2)
            _net2 = tf.keras.layers.Activation(self._activation_type)(_net2)
            _net2 = keras.layers.Conv3D(no_units, kernel_size=(3, 3, 1), padding='same', kernel_initializer=ki_resid, kernel_regularizer=tf.keras.regularizers.l2(l2_weight))(
                _net2)

            # Choose the number of gating units (either channelwise or shared) for the skip vs. 1x1x1 convs
            gating_units = 1
            if self._channelwise_gating:
                gating_units = no_units
            # Estimate a gating for the predicted change
            gating = keras.layers.Conv3D(gating_units, kernel_size=(1, 1, 1), kernel_initializer=ki_resid,
                                         activation=None)(_net2)

            def gate_convs(ips):
                skip, out, gate = ips
                gate = tf.nn.sigmoid(gate + gate_offset)
                return skip * (1.0 - gate) + out * gate

            _net2 = keras.layers.Lambda(gate_convs)([_net2_skip, _net2, gating])

            return _net, _net2

        # Make an initial 1x1x1 layer
        net1 = create_layer(self._no_units)(net)

        net2 = net1
        no_units = self._no_units
        # Create some number of convolution layers for stream 1 and gated residual blocks for stream 2
        for i in range(self._no_intermediate_layers):
            net1, net2 = create_block(net1, net2, no_units)

        no_outputs = 4
        if self._use_mvg:
            no_outputs = 5

        # Create the final layer, which produces a mean and variance for OEF and DBV
        final_layer = create_layer(no_outputs, activation=None)

        # Create an output that just looks at individual voxels
        output = final_layer(net1)

        if self._infer_inv_gamma:
            hyper_prior_layer = tfp.layers.VariableLayer(shape=(4,), dtype=tf.dtypes.float32, activation=tf.exp,
                                                   initializer=tf.keras.initializers.constant(
                                                       np.log([20.0, 2.5, 20, 2.5])))
            output = tf.concat([output, tf.ones_like(output)[:, :, :, :, :4] * hyper_prior_layer(output)], -1)

        # Create another output that also looks at neighbourhoods
        second_net = final_layer(net2)

        # Predict heteroscedastic variances
        im_sigma_layer = keras.layers.Conv3D(11, kernel_size=(1, 1, 1),
                                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=resid_init_std),
                                             bias_initializer=tf.keras.initializers.Constant(np.log(self._initial_im_sigma)),
                                             activation=tf.exp)
        im_sigma = im_sigma_layer(net2)

        # Create the model with two outputs, one with 3x3 convs for fine-tuning, and one without.
        return keras.Model(inputs=[input], outputs=[output, second_net, im_sigma])

    def scale_uncertainty(self, q_z, uncertainty_factor):
        # Scale the uncertainty by a factor (used to avoid taking massive samples)
        if self._use_mvg:
            offset = tf.reshape([0.0, tf.math.log(uncertainty_factor), 0.0, tf.math.log(uncertainty_factor), 0.0],
                                (1, 1, 1, 1, 5))
            scale = tf.reshape([1.0, 1.0, 1.0, 1.0, uncertainty_factor], (1, 1, 1, 1, 5))
            q_z = (q_z + offset) * scale
        else:
            offset = tf.reshape([0.0, tf.math.log(uncertainty_factor), 0.0, tf.math.log(uncertainty_factor)],
                                (1, 1, 1, 1, 4))
            q_z = q_z + offset

        return q_z

    def build_fine_tuner(self, encoder_model, signal_generation_layer, input_im, input_mask):
        net = input_im

        _, predicted_distribution, im_sigma = encoder_model(net)

        # Allow for multiple samples by concatenating copies
        predicted_distribution = tf.concat([predicted_distribution for x in range(self._no_samples)], 0)
        im_sigma = tf.concat([im_sigma for x in range(self._no_samples)], 0)

        sampled_oef_dbv = ReparamTrickLayer(self)((predicted_distribution, input_mask))



        if self._use_population_prior:
            if self._use_mvg:
                pop_prior = tfp.layers.VariableLayer(shape=(5,), dtype=tf.dtypes.float32,
                                                     initializer=tf.keras.initializers.constant(
                                                         [-0.97, 0.4, -1.14, 0.6, 0.0]))

                pop_prior = tf.reshape(pop_prior(predicted_distribution), (1, 1, 1, 1, 5))
            else:
                if self._mog_components > 1:
                    init = tf.keras.initializers.random_normal(stddev=1.0)
                else:
                    init = tf.keras.initializers.constant([-0.97, 0.4, -1.14, 0.6])
                pop_prior = tfp.layers.VariableLayer(shape=(4 * self._mog_components,), dtype=tf.dtypes.float32,
                                                     initializer=init)
                #
                pop_prior = tf.reshape(pop_prior(predicted_distribution), (1, 1, 1, 1, 4 * self._mog_components))

            ones_img = tf.concat([tf.ones_like(predicted_distribution) for x in range(self._mog_components)], -1)
            pop_prior_image = ones_img * pop_prior
            predicted_distribution = tf.concat([predicted_distribution, pop_prior_image], -1)

        output = signal_generation_layer(sampled_oef_dbv)

        if self._heteroscedastic_noise:
            output = tf.concat([output, im_sigma], -1)
        else:
            sigma_layer = tfp.layers.VariableLayer(shape=(1,), dtype=tf.dtypes.float32, activation=tf.exp,
                                                   initializer=tf.keras.initializers.constant(
                                                       np.log(self._initial_im_sigma)))
            output = tf.concat([output, tf.ones_like(output[:, :, :, :, 0:1]) * sigma_layer(output)], -1)


        full_model = keras.Model(inputs=[input_im, input_mask],
                                 outputs={'predictions': predicted_distribution, 'predicted_images': output})
        return full_model

    def transform_std(self, pred_stds):
        # Transform the predicted std-dev to the correct range
        return (tf.tanh(pred_stds) * 2.0) - 1.0

    def transform_offdiag(self, pred_offdiag):
        # Limit the magnitude of off-diagonal terms by pushing through a tanh
        return tf.tanh(pred_offdiag)*3.0

    def inv_transform_std(self, std):
        return tf.math.atanh((std+1.0) / 2.0)

    def forward_transform(self, logits):
        # Define the forward transform of the predicted parameters to OEF/DBV
        oef, dbv = tf.split(logits, 2, -1)
        oef = tf.nn.sigmoid(oef) * self._oef_range + self._min_oef
        dbv = tf.nn.sigmoid(dbv) * self._dbv_range + self._min_dbv
        output = tf.concat([oef, dbv], axis=-1)
        return output

    def backwards_transform(self, signal):
        # Define how to backwards transform OEF/DBV to the same parameterisation used by the NN
        oef, dbv = tf.split(signal, 2, -1)
        oef = logit((oef - self._min_oef) / self._oef_range)
        dbv = logit((dbv - self._min_dbv) / self._dbv_range)
        output = tf.concat([oef, dbv], axis=-1)
        return output

    def oef_dbv_metrics(self, y_true, y_pred, oef_dbv_r2p=0):
        """
        Produce the MSE of the predictions of OEF or DBV
        @param oef is a boolean, if False produces the output for DBV
        """
        # Reshape the data such that we can work with either volumes or single voxels
        y_true = tf.reshape(y_true, (-1, 3))
        if self._infer_inv_gamma:
            y_pred, _ = tf.split(y_pred, 2, -1)
        if self._use_mvg:
            y_pred = tf.reshape(y_pred, (-1, 5))
        else:
            y_pred = tf.reshape(y_pred, (-1, 4))

        # keras.backend.print_tensor(tf.reduce_mean(tf.exp(y_pred[:,1])))
        means = tf.stack([y_pred[:, 0], y_pred[:, 2]], -1)
        means = self.forward_transform(means)
        residual = means - y_true[:, 0:2]
        if oef_dbv_r2p == 0:
            residual = residual[:, 0]
        elif oef_dbv_r2p == 1:
            residual = residual[:, 1]
        else:
            r2p = self.calculate_r2p(means[:, 0], means[:, 1])
            residual = y_true[:, 2] - r2p

        return tf.reduce_mean(tf.square(residual))

    def oef_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 0)

    def dbv_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 1)

    def r2p_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 2)

    @staticmethod
    def squared_whitened_residual(obs, mean, oef_log_std, dbv_log_std, oef_dbv_cov):
        out_shape = tf.shape(mean)[:-1]
        obs = tf.reshape(obs, (-1, 2))
        mean = tf.reshape(mean, (-1, 2))
        oef_log_std = tf.reshape(oef_log_std, (-1, 1))
        dbv_log_std = tf.reshape(dbv_log_std, (-1, 1))
        oef_dbv_cov = tf.reshape(oef_dbv_cov, (-1, 1))

        tl = tf.exp(oef_log_std)
        br = tf.exp(dbv_log_std)
        od = oef_dbv_cov
        inv_tl = 1.0 / tl
        inv_br = 1.0 / br
        inv_bl = inv_tl * od * inv_br * -1.0
        residual = obs - mean
        whitened_residual_oef = residual[:, 0:1] * inv_tl + residual[:, 1:2] * inv_bl
        whitened_residual_dbv = residual[:, 1:2] * inv_br
        whitened_residual = tf.concat([whitened_residual_oef, whitened_residual_dbv], -1)
        squared_whitened_residual = tf.reduce_sum(tf.square(whitened_residual), -1)
        squared_whitened_residual = tf.reshape(squared_whitened_residual, out_shape)
        return squared_whitened_residual

    @staticmethod
    def calculate_chol_det(oef_log_std, dbv_log_std):
        # Calculate covariance matrix
        tl = tf.exp(oef_log_std)
        br = tf.exp(dbv_log_std)
        # Get the determinant (product of squared diagonals)
        det = tl ** 2 * br ** 2
        return det

    def synthetic_data_loss(self, y_true_orig, y_pred_orig, use_r2p_loss, inv_gamma_alpha, inv_gamma_beta):
        # Reshape the data such that we can work with either volumes or single voxels
        y_true_orig = tf.reshape(y_true_orig, (-1, 3))
        # Backwards transform the true values (so we can define our distribution in the parameter space)
        y_true = self.backwards_transform(y_true_orig[:, 0:2])
        if self._infer_inv_gamma:
            y_pred_orig, inv_gamma_params = tf.split(y_pred_orig, 2, axis=-1)

        if self._use_mvg:
            y_pred = tf.reshape(y_pred_orig, (-1, 5))
        else:
            y_pred = tf.reshape(y_pred_orig, (-1, 4))

        oef_mean = y_pred[:, 0]
        oef_log_std = self.transform_std(y_pred[:, 1])
        dbv_mean = y_pred[:, 2]
        dbv_log_std = self.transform_std(y_pred[:, 3])

        def gaussian_nll_chol(obs, mean, oef_log_std, oef_dbv_cov, dbv_log_std):
            det = EncoderTrainer.calculate_chol_det(oef_log_std, dbv_log_std)
            # Calculate the inverse cholesky matrix
            squared_whitened_residual = EncoderTrainer.squared_whitened_residual(obs, mean, oef_log_std, dbv_log_std, oef_dbv_cov)
            return -(-0.5*tf.math.log(det)-0.5 * squared_whitened_residual)

        def gaussian_nll(obs, mean, log_std):
            return -(-log_std - (1.0 / 2.0) * ((obs - mean) / tf.exp(log_std)) ** 2)
        if self._use_mvg:
            loss = gaussian_nll_chol(y_true[:, :2], tf.stack([oef_mean, dbv_mean], -1), oef_log_std,
                                     self.transform_offdiag(y_pred[:, 4]), dbv_log_std)
        else:
            # Gaussian negative log likelihoods
            oef_nll = gaussian_nll(y_true[:, 0], oef_mean, oef_log_std)
            dbv_nll = gaussian_nll(y_true[:, 1], dbv_mean, dbv_log_std)


            loss = oef_nll + dbv_nll

        """hct_mean = y_pred[:, 4]
        hct_log_std = transform_std(y_pred[:, 5])"""
        if use_r2p_loss:
            # Could use sampling to calculate the distribution on r2p - need to forward transform the oef/dbv parameters
            rpl = ReparamTrickLayer(self)
            predictions = []
            n_samples = 10
            for i in range(n_samples):
                predictions.append(rpl([y_pred_orig, tf.ones_like(y_pred_orig[:, :, :, :, 0:1])]))

            predictions = tf.stack(predictions, -1)
            predictions = tf.reshape(predictions, (-1, 2, n_samples))
            r2p = self.calculate_r2p(predictions[:, 0, :], predictions[:, 1, :])
            # Calculate a normal distribution for r2 prime from these samples
            r2p_mean = tf.reduce_mean(r2p, -1)
            r2p_log_std = tf.math.log(tf.math.reduce_std(r2p, -1))
            r2p_nll = gaussian_nll(y_true_orig[:, 2], r2p_mean, r2p_log_std)
            loss = loss + r2p_nll

        if (inv_gamma_alpha * inv_gamma_beta > 0.0) or self._infer_inv_gamma:
            if self._infer_inv_gamma:
                inv_gamma_params = inv_gamma_params[0,0,0,0,:]
                inv_gamma_oef = tfp.distributions.InverseGamma(inv_gamma_params[0], inv_gamma_params[1])
                inv_gamma_dbv = tfp.distributions.InverseGamma(inv_gamma_params[2], inv_gamma_params[3])
            else:
                inv_gamma_oef = inv_gamma_dbv = tfp.distributions.InverseGamma(inv_gamma_alpha, inv_gamma_beta)
            if self._use_mvg:
                oef_var = tf.exp(oef_log_std)**2
                dbv_var = tf.exp(dbv_log_std)**2 + y_pred[:, 4]**2
            else:
                oef_var = tf.exp(oef_log_std * 2.0)
                dbv_var = tf.exp(dbv_log_std * 2.0)
            prior_loss = inv_gamma_oef.log_prob(oef_var)
            prior_loss = prior_loss + inv_gamma_dbv.log_prob(dbv_var)
            loss = loss - prior_loss

        """ig = tfp.distributions.InverseGamma(3, 0.15)
        lp_oef = ig.log_prob(tf.exp(oef_log_std*2))
        lp_dbv = ig.log_prob(tf.exp(dbv_log_std*2))
        nll = nll - (lp_oef + lp_dbv) """

        return tf.reduce_mean(loss)

    def calculate_dw(self, oef):
        from signals import SignalGenerationLayer
        dchi = float(self._system_params['dchi'])
        b0 = float(self._system_params['b0'])
        gamma = float(self._system_params['gamma'])
        hct = float(self._system_params['hct'])
        return SignalGenerationLayer.calculate_dw_static(oef, hct, gamma, b0, dchi)

    def calculate_r2p(self, oef, dbv):
        return self.calculate_dw(oef) * dbv

    def fine_tune_loss_fn(self, y_true, y_pred, return_mean=True):
        # Deal with multiple samples by concatenating
        y_true = tf.concat([y_true for x in range(self._no_samples)], 0)
        mask = y_true[:, :, :, :, -1:]
        if self._heteroscedastic_noise:
            y_pred, sigma = tf.split(y_pred, 2, -1)
            sigma = tf.reshape(sigma, (-1, 11))
        else:
            sigma = tf.reduce_mean(y_pred[:, :, :, :, -1:])
            y_pred = y_pred[:, :, :, :, :-1]

        # Normalise and mask the predictions/real data
        if self._multi_image_normalisation:
            y_true = y_true / (tf.reduce_mean(y_true[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
            y_pred = y_pred / (tf.reduce_mean(y_pred[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
        else:
            y_true = y_true / (tf.reduce_mean(y_true[:, :, :, :, 2:3], -1, keepdims=True) + 1e-3)
            y_pred = y_pred / (tf.reduce_mean(y_pred[:, :, :, :, 2:3], -1, keepdims=True) + 1e-3)

        if self._predict_log_data:
            y_true = tf.where(mask > 0, tf.math.log(y_true), tf.zeros_like(y_true))
            y_pred = tf.where(mask > 0, tf.math.log(y_pred), tf.zeros_like(y_pred))

        # Calculate the residual difference between our normalised data
        residual = y_true[:, :, :, :, :-1] - y_pred
        residual = tf.reshape(residual, (-1, 11))
        mask = tf.reshape(mask, (-1, 1))

        # Optionally use a student-t distribution (with heavy tails) or a Gaussian distribution if dof >= 50
        if self._student_t_df is not None and self._student_t_df < 50:
            dist = tfp.distributions.StudentT(df=self._student_t_df, loc=0.0, scale=sigma)
            nll = -dist.log_prob(residual)
        else:
            nll = -(-tf.math.log(sigma) - np.log(np.sqrt(2.0 * np.pi)) - 0.5 * tf.square(residual / sigma))

        nll = tf.reduce_sum(nll, -1, keepdims=True)
        nll = nll * mask
        if return_mean:
            return tf.reduce_sum(nll) / tf.reduce_sum(mask)
        else:
            return nll

    def kl_loss(self, true, predicted, return_mean=True):

        true = tf.concat([true for x in range(self._no_samples)], 0)
        prior_cost = 0.0
        if self._use_mvg:
            if self._use_population_prior:

                q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, q_oef_dbv_cov, p_oef_mean, p_oef_log_std, \
                p_dbv_mean, p_dbv_log_std, p_oef_dbv_cov = tf.split(predicted, 10, -1)
                _, _, _, _, _, mask = tf.split(true, 6, -1)
            else:
                q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, q_oef_dbv_cov = tf.split(predicted, 5, -1)
                p_oef_mean, p_oef_log_std, p_dbv_mean, p_dbv_log_std, p_oef_dbv_cov, mask = tf.split(true, 6, -1)

            q_oef_dbv_cov = self.transform_offdiag(q_oef_dbv_cov)
            p_oef_dbv_cov = self.transform_offdiag(p_oef_dbv_cov)
            q_oef_log_std, q_oef_log_std = tf.split(self.transform_std(tf.concat([q_oef_log_std, q_oef_log_std], -1)), 2, -1)
            p_oef_log_std, p_oef_log_std = tf.split(self.transform_std(tf.concat([p_oef_log_std, p_oef_log_std], -1)),
                                                    2, -1)
            det_q = EncoderTrainer.calculate_chol_det(q_oef_log_std, q_dbv_log_std)
            det_p = EncoderTrainer.calculate_chol_det(p_oef_log_std, p_dbv_log_std)
            p_mu = tf.concat([p_oef_mean, p_dbv_mean], -1)
            q_mu = tf.concat([q_oef_mean, q_dbv_mean], -1)

            squared_residual = EncoderTrainer.squared_whitened_residual(p_mu, q_mu, p_oef_log_std, p_dbv_log_std, p_oef_dbv_cov)
            det_term = tf.math.log(det_p) - tf.math.log(det_q)

            inv_p_tl = 1.0/tf.exp(p_oef_log_std)
            inv_p_br = 1.0/tf.exp(p_dbv_log_std)
            inv_p_od = inv_p_tl * p_oef_dbv_cov * inv_p_br * -1.0
            inv_pcov_tl = inv_p_tl**2
            inv_pcov_br = inv_p_od**2 + inv_p_br**2
            inv_pcov_od = inv_p_tl*inv_p_od

            q_tl = tf.exp(q_oef_log_std)**2
            q_br = tf.exp(q_dbv_log_std)**2 + q_oef_dbv_cov**2
            q_od = q_oef_dbv_cov*tf.exp(q_oef_log_std)

            trace = inv_pcov_tl*q_tl + inv_pcov_od*q_od + inv_pcov_od*q_od + q_br*inv_pcov_br
            squared_residual = tf.expand_dims(squared_residual, -1)

            kl_op = 0.5 * (trace + squared_residual + det_term - 2.0)

            kl_op = tf.where(mask > 0, kl_op, tf.zeros_like(kl_op))
            if return_mean:
                return tf.reduce_sum(kl_op) / tf.reduce_sum(mask)
            else:
                return kl_op
        elif self._use_population_prior and (self._mog_components > 1):
                _, _, _, _, mask = tf.split(true, 5, -1)
                distributions = tf.split(predicted, self._mog_components+1, -1)
                q_distribution = distributions[0]
                entropy = self.transform_std(q_distribution[:, :, :, :, 1]) + self.transform_std(q_distribution[:, :, :, :, 3])
                oef_sample = q_distribution[:, :, :, :, 0] + tf.random.normal(tf.shape(q_distribution[:, :, :, :, 0])) * tf.exp(
                    self.transform_std(q_distribution[:, :, :, :, 1]))
                dbv_sample = q_distribution[:, :, :, :, 2] + tf.random.normal(tf.shape(q_distribution[:, :, :, :, 0])) * tf.exp(
                    self.transform_std(q_distribution[:, :, :, :, 3]))

                def gaussian_nll(sample, mean, log_std):
                    return -(-self.transform_std(log_std) - (1.0 / 2.0) * ((sample - mean) / tf.exp(self.transform_std(log_std))) ** 2)
                kl_op = entropy * -1
                for i in range(self._mog_components):
                    kl_op = kl_op + gaussian_nll(oef_sample, distributions[i+1][:, :, :, :, 0],
                                                 distributions[i+1][:, :, :, :, 1]) / float(self._mog_components)
                    kl_op = kl_op + gaussian_nll(dbv_sample, distributions[i + 1][:, :, :, :, 2],
                                                 distributions[i + 1][:, :, :, :, 3]) / float(self._mog_components)
                kl_op = tf.expand_dims(kl_op, -1)
        else:
            if self._use_population_prior:
                q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, p_oef_mean, p_oef_log_std, \
                p_dbv_mean, p_dbv_log_std = tf.split(predicted, 8, -1)
                _, _, _, _, mask = tf.split(true, 5, -1)
            else:
                # Calculate the kullback-leibler divergence between the posterior and prior distibutions
                q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std = tf.split(predicted, 4, -1)
                p_oef_mean, p_oef_log_std, p_dbv_mean, p_dbv_log_std, mask = tf.split(true, 5, -1)

            def kl(q_mean, q_log_std, p_mean, p_log_std):
                result = tf.exp(q_log_std * 2 - p_log_std * 2) + tf.square(p_mean - q_mean) * tf.exp(p_log_std * -2.0)
                result = result + p_log_std * 2 - q_log_std * 2 - 1.0
                return result * 0.5

            q_oef_log_std, q_dbv_log_std = tf.split(self.transform_std(tf.concat([q_oef_log_std, q_dbv_log_std], -1)), 2,
                                                    -1)
            p_oef_log_std, p_dbv_log_std = tf.split(self.transform_std(tf.concat([p_oef_log_std, p_dbv_log_std], -1)),
                                                    2, -1)
            kl_oef = kl(q_oef_mean, q_oef_log_std, p_oef_mean, p_oef_log_std)
            kl_dbv = kl(q_dbv_mean, q_dbv_log_std, p_dbv_mean, p_dbv_log_std)
            # Mask the KL
            kl_op = (kl_oef + kl_dbv)

            if self._use_population_prior:
                ig_dist = tfp.distributions.InverseGamma(1.0, 2.0)
                prior_cost = -ig_dist.log_prob(tf.exp(tf.reduce_mean(p_dbv_log_std)*2.0))
                prior_cost = prior_cost - ig_dist.log_prob(tf.exp(tf.reduce_mean(p_oef_log_std) * 2.0))
                prior_cost = prior_cost * tf.cast(tf.shape(predicted)[0], tf.float32)

        kl_op = tf.where(mask > 0, kl_op, tf.zeros_like(kl_op))

        # keras.backend.print_tensor(kl_op)
        if return_mean:
            return (tf.reduce_sum(kl_op) + prior_cost) / tf.reduce_sum(mask)
        else:
            return kl_op

    def smoothness_loss(self, true_params, pred_params):
        true_params = tf.concat([true_params for x in range(self._no_samples)], 0)
        # Define a total variation smoothness term for the predicted means
        if self._use_mvg:
            q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, _ = tf.split(pred_params, 5, -1)
            _, _, _, _, _, mask = tf.split(true_params, 6, -1)
        else:
            q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std = tf.split(pred_params, 4, -1)
            _, _, _, _, mask = tf.split(true_params, 5, -1)
        pred_params = tf.concat([q_oef_mean, q_dbv_mean], -1)
        # Forward transform the parameters to OEF/DBV space rather than logits
        pred_params = self.forward_transform(pred_params)
        # Rescale the range
        pred_params = pred_params / tf.reshape([self._oef_range, self._dbv_range], (1, 1, 1, 1, 2))


        diff_x = pred_params[:, :-1, :, :, :] - pred_params[:, 1:, :, :, :]
        x_mask = tf.logical_and(mask[:, :-1, :, :, :] > 0.0, mask[:, 1:, :, :, :] > 0.0)
        diff_x = tf.where(x_mask, diff_x, tf.zeros_like(diff_x))

        diff_y = pred_params[:, :, :-1, :, :] - pred_params[:, :, 1:, :, :]
        y_mask = tf.logical_and(mask[:, :, :-1, :, :] > 0.0, mask[:, :, 1:, :, :] > 0.0)
        diff_y = tf.where(y_mask, diff_y, tf.zeros_like(diff_y))

        #diff_z = pred_params[:, :, :, :-1, :] - pred_params[:, :, :, 1:, :]
        #diffs = tf.reduce_mean(tf.abs(diff_x)) + tf.reduce_mean(tf.abs(diff_y))
        diffs = tf.reduce_sum(tf.abs(diff_x)) + tf.reduce_sum(tf.abs(diff_y))  # + tf.reduce_mean(tf.abs(diff_z))
        diffs = diffs / tf.reduce_sum(mask)
        return diffs

    def estimate_population_param_distribution(self, model, data):
        _, predictions, _ = model.predict(data[:, :, :, :, :-1] * data[:, :, :, :, -1:])
        mask = data[:, :, :, :, -1:]
        oef = predictions[:, :, :, :, 0:1] * mask
        dbv = predictions[:, :, :, :, 2:3] * mask

        mask_pix = tf.reduce_sum(mask)
        mean_oef = tf.reduce_sum(oef) / mask_pix
        std_oef = tf.sqrt(tf.reduce_sum(tf.square(oef - mean_oef)*mask) / mask_pix)
        log_std_oef = self.inv_transform_std(tf.math.log(std_oef))
        mean_dbv = tf.reduce_sum(dbv) / mask_pix
        std_dbv = tf.sqrt(tf.reduce_sum(tf.square(dbv - mean_dbv) * mask) / mask_pix)
        log_std_dbv = self.inv_transform_std(tf.math.log(std_dbv))
        print(mean_oef, log_std_oef, mean_dbv, log_std_dbv)

    def save_predictions(self, model, data, filename, transform_directory=None, use_first_op=True, fine_tuner_model=None,
                         priors=None):
        import nibabel as nib

        predictions, predictions2, predicted_im_sigma = model.predict(data[:, :, :, :, :-1] * data[:, :, :, :, -1:])
        if use_first_op is False:
            predictions = predictions2
        elif self._infer_inv_gamma:
            predictions, inv_gamma_params = tf.split(predictions, 2, axis=-1)

        # Get the log stds, but don't transform them. Their meaning is complicated because of the forward transformation
        log_stds = tf.concat([predictions[:, :, :, :, 1:2], predictions[:, :, :, :, 3:4]], -1)
        log_stds = self.transform_std(log_stds)
        if self._use_mvg:
            log_stds = tf.concat([log_stds, self.transform_offdiag(predictions[:, :, :, :, 4:5])], -1)

        # Extract the OEF and DBV and transform them
        predictions = tf.concat([predictions[:, :, :, :, 0:1], predictions[:, :, :, :, 2:3]], -1)

        predictions = self.forward_transform(predictions)

        def save_im_data(im_data, _filename):
            existing_nib = nib.load(transform_directory + '/example.nii.gz')
            new_header = existing_nib.header.copy()
            images = np.split(im_data, im_data.shape[0], axis=0)
            images = np.squeeze(np.concatenate(images, axis=-1), 0)
            array_img = nib.Nifti1Image(images, None, header=new_header)

            nib.save(array_img, _filename + '.nii.gz')

        oef = predictions[:, :, :, :, 0:1]
        dbv = predictions[:, :, :, :, 1:2]
        r2p = self.calculate_r2p(oef, dbv)

        if fine_tuner_model:
            data = np.float32(data)
            outputs = fine_tuner_model.predict([data[:, :, :, :, :-1], data[:, :, :, :, -1:]])
            pred_dist = outputs['predictions']
            y_pred = outputs['predicted_images']
            mask = data[:, :, :, :, -1:]
            y_true = data[:, :, :, :, :-1]
            likelihood_map = self.fine_tune_loss_fn(data, y_pred, return_mean=False)
            if self._use_population_prior:
                dists = tf.split(pred_dist, self._mog_components+1, -1)
                priors = dists[0]

            kl_map = self.kl_loss(np.concatenate([priors, mask], -1), pred_dist, return_mean=False)
            likelihood_map = np.reshape(likelihood_map, data.shape[0:4]+(1,))
            save_im_data(likelihood_map, filename + '_likelihood')

            if data.shape[-1] == 5:
                kl_map = np.reshape(kl_map, data.shape[0:4]+(1,))
            elif self._use_population_prior == False:
                kl_map = np.reshape(kl_map, data.shape[0:4] + (1,))
            save_im_data(kl_map, filename + '_kl')
            #y_pred, predicted_sigma = tf.split(y_pred, 2, -1)
            y_pred = y_pred[:, :, :, :, :11]
            if self._multi_image_normalisation:
                y_true = y_true / (np.mean(y_true[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
                y_pred = y_pred / (np.mean(y_pred[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
            else:
                y_true = y_true / (np.mean(y_true[:, :, :, :, 2:3], -1, keepdims=True) + 1e-3)
                y_pred = y_pred / (np.mean(y_pred[:, :, :, :, 2:3], -1, keepdims=True) + 1e-3)

            residual = np.mean(tf.abs(y_true - y_pred), -1, keepdims=True)
            save_im_data(residual, filename + '_residual')

        if transform_directory:
            import os
            mni_ims = filename + '_merged.nii.gz'
            merge_cmd = 'fslmerge -t ' + mni_ims
            ref_image = transform_directory + '/MNI152_T1_2mm.nii.gz'
            for i in range(oef.shape[0]):
                nonlin_transform = transform_directory + '/nonlin' + str(i) + '.nii.gz'
                oef_im = oef[i, ...]
                dbv_im = dbv[i, ...]
                r2p_im = r2p[i, ...]
                subj_ims = np.stack([oef_im, dbv_im, r2p_im], 0)

                subj_im = filename + '_subj' + str(i)
                save_im_data(subj_ims, subj_im)
                subj_im_mni = subj_im + 'mni'
                # Transform
                cmd = 'applywarp --in=' + subj_im + ' --out=' + subj_im_mni + ' --warp=' + nonlin_transform + \
                      ' --ref=' + ref_image
                os.system(cmd)
                merge_cmd = merge_cmd + ' ' + subj_im_mni

            os.system(merge_cmd)
            merged_nib = nib.load(mni_ims)
            merged_data = merged_nib.get_fdata()

            file_types = ['_oef_mni', '_dbv_mni', '_r2p_mni']
            for t_idx, t in enumerate(file_types):
                t_data = merged_data[:, :, :, t_idx::3]
                new_header = merged_nib.header.copy()
                array_img = nib.Nifti1Image(t_data, affine=None, header=new_header)
                nib.save(array_img, filename + t + '.nii.gz')


        save_im_data(oef, filename + '_oef')
        save_im_data(dbv, filename + '_dbv')
        save_im_data(r2p, filename + '_r2p')

        # save_im_data(predictions[:, :, :, :, 2:3], filename + '_hct')
        save_im_data(log_stds, filename + '_logstds')
