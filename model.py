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
        self._dbv_range = 0.2
        self._min_dbv = 0.001
        self._heteroscedastic_noise = heteroscedastic_noise
        self._predict_log_data = predict_log_data
        # Store the spin-echo index
        self._se_idx = 0#int(abs(float(system_params['tau_start'])/float(system_params['tau_step'])))

    def normalise_data(self, _data):
        # Do the normalisation as part of the model rather than as pre-processing
        orig_shape = tf.shape(_data)
        _data = tf.reshape(_data, (-1, _data.shape[-1]))
        _data = tf.clip_by_value(_data, 1e-2, 1e8)
        if self._multi_image_normalisation:
            # Normalise based on the mean of tau =0 and adjacent tau values to minimise the effects of noise
            _data = _data / tf.reduce_mean(_data[:, self._se_idx-1:self._se_idx+2], -1, keepdims=True)
        else:
            _data = _data / tf.reduce_mean(_data[:, self._se_idx:self._se_idx+1], -1, keepdims=True)
        # Take the logarithm
        log_data = tf.math.log(_data)
        log_data = tf.reshape(log_data, orig_shape)

        # _data = tf.reshape(_data, orig_shape)
        # return tf.concat([_data, log_data], -1)
        return log_data

    def create_layer(self, _no_units, activation='default'):
        # if activation is 'default', use the value in the class. 
        if activation == 'default':
            activation = self._activation_type
        ki = tf.keras.initializers.HeNormal()
        return keras.layers.Conv3D(_no_units, kernel_size=(1, 1, 1), kernel_initializer=ki, activation=activation)

    def create_encoder(self, system_constants=None, gate_offset=0.0, resid_init_std=1e-1, no_ip_images=11):
        """
        @params: system_constants (array): If not None, perform a dense transformation and multiply with first level representation
        @params: gate_offset (float): How much to offset the initial gating towards the MLP
        @params: resid_init_std (float): std dev for weights in the residual gated network
        """

        ki_resid = tf.keras.initializers.RandomNormal(stddev=resid_init_std)



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
            conv_layer = self.create_layer(no_units)
            _net = conv_layer(_net_in)

            # Apply the same 1x1x1 conv as for stream 1 for the skip connection
            _net2_skip = conv_layer(_net2_in)
            # Do a residual block
            _net2 = add_normalizer(_net2_in)
            _net2 = tf.keras.layers.Activation(self._activation_type)(_net2)
            _net2 = keras.layers.Conv3D(no_units, kernel_size=(3, 3, 1), padding='same', kernel_initializer=ki_resid)(
                _net2)
            _net2 = add_normalizer(_net2)
            _net2 = tf.keras.layers.Activation(self._activation_type)(_net2)
            _net2 = keras.layers.Conv3D(no_units, kernel_size=(3, 3, 1), padding='same', kernel_initializer=ki_resid)(
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

        input_layer = keras.layers.Input(shape=(None, None, None, no_ip_images), ragged=False)

        first_conv_ip = keras.layers.Lambda(self.normalise_data)(input_layer)

        # Make an initial 1x1x1 layer
        first_conv_op = self.create_layer(self._no_units)(first_conv_ip)
        # Make an input following the initial layer, to be used for transferring to different data
        after_first_conv_input = keras.layers.Input(shape=(None, None, None, self._no_units), ragged=False)

        net2 = net1 = after_first_conv_input
        no_units = self._no_units
        # Create some number of convolution layers for stream 1 and gated residual blocks for stream 2
        for i in range(self._no_intermediate_layers):
            net1, net2 = create_block(net1, net2, no_units)

        no_outputs = 4
        if self._use_mvg:
            no_outputs = 5

        # Create the final layer, which produces a mean and variance for OEF and DBV
        final_layer = self.create_layer(no_outputs, activation=None)

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
        im_sigma_layer = keras.layers.Conv3D(no_ip_images, kernel_size=(1, 1, 1),
                                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=resid_init_std),
                                             bias_initializer=tf.keras.initializers.Constant(np.log(self._initial_im_sigma)),
                                             activation=tf.exp)

        # Create the inner model with two outputs, one with 3x3 convs for fine-tuning, and one without.
        inner_model = keras.Model(inputs=[after_first_conv_input], outputs=[output, second_net, net2])

        _output, _second_net, _net2 = inner_model(first_conv_op)
        sigma_pred = im_sigma_layer(_net2)
        # The outer model, calls the inner model but with an initial convolution from the input data
        outer_model = keras.Model(inputs=[input_layer], outputs=[_output, _second_net, sigma_pred])
        return outer_model, inner_model

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
        return (tf.tanh(pred_stds) * 3.0) - 1.0

    def transform_offdiag(self, pred_offdiag):
        # Limit the magnitude of off-diagonal terms by pushing through a tanh
        return tf.tanh(pred_offdiag)*np.exp(-2.0)

    def inv_transform_std(self, std):
        return tf.math.atanh((std+1.0) / 3.0)

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
        oef = (oef - self._min_oef) / (self._oef_range)
        dbv = (dbv - self._min_dbv) / (self._dbv_range)
        if include_logit:
            oef = logit(oef)
            dbv = logit(dbv)
        output = tf.concat([oef, dbv], axis=-1)
        return output

    def create_samples(self, predicted_params, mask, no_samples):
        rpl = ReparamTrickLayer(self)
        samples = []
        for i in range(no_samples):
            samples.append(rpl([predicted_params, mask]))
        samples = tf.stack(samples, -1)
        return samples

    def calculate_means(self, predicted_params, mask, include_r2p=False, return_stds=False, no_samples=20):
        # Calculate the means of the logit normal variables via sampling
        samples = self.create_samples(predicted_params, mask, no_samples)
        means = tf.reduce_mean(samples, -1)
        if return_stds:
            stds = tf.reduce_mean(tf.square(samples - tf.expand_dims(means, -1)), -1)
        if include_r2p:
            r2ps = self.calculate_r2p(samples[:, :, :, :, 0, :], samples[:, :, :, :, 1, :])
            r2p_mean = tf.reduce_mean(r2ps, -1, keepdims=True)
            means = tf.concat([means, r2p_mean], -1)
            if return_stds:
                r2p_std = tf.reduce_mean(tf.square(r2ps - r2p_mean), -1, keepdims=True)
                stds = tf.concat([stds, r2p_std], -1)

        if return_stds:
            return means, stds
        else:
            return means

    def oef_dbv_metrics(self, y_true, y_pred, oef_dbv_r2p=0):
        """
        Produce the MSE of the predictions of OEF or DBV
        @param oef is a boolean, if False produces the output for DBV
        """

        means = self.calculate_means(y_pred, tf.ones_like(y_pred[:, :, :, :, 0:1]), include_r2p=True)

        # Reshape the data such that we can work with either volumes or single voxels
        y_true = tf.reshape(y_true, (-1, 3))
        means = tf.reshape(means, (-1, 3))

        residual = means - y_true
        if oef_dbv_r2p == 0:
            residual = residual[:, 0]
        elif oef_dbv_r2p == 1:
            residual = residual[:, 1]
        else:
            residual = residual[:, 2]

        return tf.reduce_mean(tf.square(residual))

    def oef_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 0)

    def dbv_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 1)

    def r2p_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 2)

    def logit_gaussian_mvg_log_prob(self, observations, predicted_params):
        original_shape = tf.shape(predicted_params)[0:4]
        # Convert our predicted parameters
        predicted_params = tf.reshape(predicted_params, (-1, 5))
        oef_mean = predicted_params[:, 0]
        oef_log_std = self.transform_std(predicted_params[:, 1])
        dbv_mean = predicted_params[:, 2]
        dbv_log_std = self.transform_std(predicted_params[:, 3])

        def gaussian_nll_chol(obs, mean, oef_log_std, oef_dbv_cov, dbv_log_std):
            log_det = EncoderTrainer.calculate_log_chol_det(oef_log_std, dbv_log_std)
            # Calculate the inverse cholesky matrix
            squared_whitened_residual = EncoderTrainer.squared_whitened_residual(obs, mean, oef_log_std,
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

    @staticmethod
    def gaussian_nll(obs, mean, log_std):
        return -(-log_std - (1.0 / 2.0) * ((obs - mean) / tf.exp(log_std)) ** 2)

    def logit_gaussian_log_prob(self, observations, predicted_params):
        original_shape = tf.shape(predicted_params)[0:4]
        # Convert our predicted parameters
        predicted_params = tf.reshape(predicted_params, (-1, 4))
        oef_mean = predicted_params[:, 0]
        oef_log_std = self.transform_std(predicted_params[:, 1])
        dbv_mean = predicted_params[:, 2]
        dbv_log_std = self.transform_std(predicted_params[:, 3])

        # Scale our observation to 0-1 range
        x = self.backwards_transform(observations[:, 0:2], False)
        loss_oef = self.gaussian_nll(logit(x[:, 0]), oef_mean, oef_log_std)
        loss_dbv = self.gaussian_nll(logit(x[:, 1]), dbv_mean, dbv_log_std)
        loss = loss_oef + loss_dbv + tf.reduce_sum(tf.math.log(x*(1.0-x)), -1)
        loss = tf.reshape(loss, original_shape)
        return loss

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
        inv_bl = tf.exp(oef_log_std*-1.0 + dbv_log_std*-1.0) * oef_dbv_cov * -1.0
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

    def synthetic_data_loss(self, y_true_orig, y_pred_orig, use_r2p_loss, inv_gamma_alpha, inv_gamma_beta):
        # Reshape the data such that we can work with either volumes or single voxels
        y_true_orig = tf.reshape(y_true_orig, (-1, 3))
        # Backwards transform the true values (so we can define our distribution in the parameter space)
        y_true = self.backwards_transform(y_true_orig[:, 0:2], True)
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


        if self._use_mvg:
            loss = self.logit_gaussian_mvg_log_prob(y_true_orig[:, :2], y_pred_orig)
        else:
            loss = self.logit_gaussian_log_prob(y_true_orig[:, :2], y_pred_orig)

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
            r2p_nll = self.gaussian_nll(y_true_orig[:, 2], r2p_mean, r2p_log_std)
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
        no_images = y_true.shape[-1]-1
        if self._heteroscedastic_noise:
            y_pred, sigma = tf.split(y_pred, 2, -1)
            sigma = tf.reshape(sigma, (-1, no_images))
        else:
            sigma = tf.reduce_mean(y_pred[:, :, :, :, -1:])
            y_pred = y_pred[:, :, :, :, :-1]

        # Normalise and mask the predictions/real data
        if self._multi_image_normalisation:
            y_true = y_true / (tf.reduce_mean(y_true[:, :, :, :, self._se_idx-1:self._se_idx+2], -1, keepdims=True) + 1e-3)
            y_pred = y_pred / (tf.reduce_mean(y_pred[:, :, :, :, self._se_idx-1:self._se_idx+2], -1, keepdims=True) + 1e-3)
        else:
            y_true = y_true / (tf.reduce_mean(y_true[:, :, :, :, self._se_idx:self._se_idx+1], -1, keepdims=True) + 1e-3)
            y_pred = y_pred / (tf.reduce_mean(y_pred[:, :, :, :, self._se_idx:self._se_idx+1], -1, keepdims=True) + 1e-3)

        if self._predict_log_data:
            y_true = tf.where(mask > 0, tf.math.log(y_true), tf.zeros_like(y_true))
            y_pred = tf.where(mask > 0, tf.math.log(y_pred), tf.zeros_like(y_pred))

        # Calculate the residual difference between our normalised data
        residual = y_true[:, :, :, :, :-1] - y_pred
        residual = tf.reshape(residual, (-1, no_images))
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

    def mvg_kl_analytic(self, prior, pred):
        q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std, q_oef_dbv_cov = tf.split(pred, 5, -1)
        p_oef_mean, p_oef_log_std, p_dbv_mean, p_dbv_log_std, p_oef_dbv_cov, mask = tf.split(prior, 6, -1)

        q_oef_log_std, q_dbv_log_std = tf.split(self.transform_std(tf.concat([q_oef_log_std, q_dbv_log_std], -1)), 2, -1)
        p_oef_log_std, p_dbv_log_std = tf.split(self.transform_std(tf.concat([p_oef_log_std, p_dbv_log_std], -1)), 2, -1)
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
        return kl_oef+kl_dbv

    def mvg_kl_samples(self, prior, pred, no_samples=50):
        prior_dist, mask = tf.split(prior, [5, 1], -1)
        samples = self.create_samples(pred, mask, no_samples)

        log_q = [-self.logit_gaussian_mvg_log_prob(tf.reshape(samples[:, :, :, :, :, x], (-1, 2)), tf.stop_gradient(pred)) for x in range(no_samples)]
        log_p = [-self.logit_gaussian_mvg_log_prob(tf.reshape(samples[:, :, :, :, :, x], (-1, 2)), prior_dist) for x in range(no_samples)]

        log_q = tf.stack(log_q, -1)
        log_p = tf.stack(log_p, -1)

        #finite_mask = tf.logical_and(tf.math.is_finite(log_q), tf.math.is_finite(log_p))
        kl_op = log_q - log_p
        """
        tf.keras.backend.print_tensor(tf.reduce_mean(kl_op))
        kl_op = tf.where(tf.math.is_finite(kl_op), kl_op, tf.zeros_like(kl_op))
        kl_op = tf.reduce_sum(kl_op, -1, keepdims=True) / tf.reduce_sum(tf.cast(tf.math.is_finite(kl_op), tf.float32), -1, keepdims=True)
        """
        kl_op = tf.reduce_mean(kl_op, axis=-1, keepdims=True)
        return kl_op

    def mvg_kl(self, true, predicted):
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

    def kl_loss(self, true, predicted, return_mean=True, no_samples=70):

        true = tf.concat([true for x in range(self._no_samples)], 0)
        prior_cost = 0.0
        if self._use_mvg:
            kl_op = self.mvg_kl_samples(true, predicted, no_samples=no_samples)
            _, mask = tf.split(true, [5, 1], -1)
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
                q = tfp.distributions.LogitNormal(loc=q_mean, scale=tf.exp(q_log_std))
                p = tfp.distributions.LogitNormal(loc=p_mean, scale=tf.exp(p_log_std))
                return q.kl_divergence(p)
                """result = tf.exp(q_log_std * 2 - p_log_std * 2) + tf.square(p_mean - q_mean) * tf.exp(p_log_std * -2.0)
                result = result + p_log_std * 2 - q_log_std * 2 - 1.0
                return result * 0.5"""

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
        print('final results for mean_oef, log_std_oef, mean_dbv, log_std_dbv, respectively: ')
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
        #predictions = tf.concat([predictions[:, :, :, :, 0:1], predictions[:, :, :, :, 2:3]], -1)
        means, log_stds = self.calculate_means(predictions, tf.ones_like(predictions[:, :, :, :, :1]), include_r2p=True, return_stds=True, no_samples=200)

        def save_im_data(im_data, _filename):
            images = np.split(im_data, im_data.shape[0], axis=0)
            images = np.squeeze(np.concatenate(images, axis=-1), 0)
            if transform_directory is not None:
                existing_nib = nib.load(transform_directory + '/example.nii.gz')
                new_header = existing_nib.header.copy()
                array_img = nib.Nifti1Image(images, None, header=new_header)
            else:
                array_img = nib.Nifti1Image(images, None)

            nib.save(array_img, _filename + '.nii.gz')
        oef, dbv, r2p = tf.split(means, 3, -1)

        if fine_tuner_model:
            data = np.float32(data)

            likelihood_map_list = []
            no_samples = 100
            for i in range(no_samples):
                outputs = fine_tuner_model.predict([data[:, :, :, :, :-1], data[:, :, :, :, -1:]])
                pred_dist = outputs['predictions']
                y_pred = outputs['predicted_images']
                likelihood_map_tmp = self.fine_tune_loss_fn(data, y_pred, return_mean=False)
                likelihood_map_list.append(likelihood_map_tmp)

            ave_likelihood_map = tf.reduce_mean(tf.stack(likelihood_map_list, -1), -1)
            # Extract the mask
            mask = data[:, :, :, :, -1:]
            # Remove the mask from y_true
            y_true = data[:, :, :, :, :-1]

            if self._use_population_prior:
                dists = tf.split(pred_dist, self._mog_components+1, -1)
                priors = dists[0]

            kl_map = self.kl_loss(np.concatenate([priors, mask], -1), pred_dist, return_mean=False, no_samples=100)
            ave_likelihood_map = np.reshape(ave_likelihood_map, data.shape[0:4]+(1,))
            save_im_data(ave_likelihood_map, filename + '_likelihood')

            if data.shape[-1] == 5:
                kl_map = np.reshape(kl_map, data.shape[0:4]+(1,))
            elif self._use_population_prior == False:
                kl_map = np.reshape(kl_map, data.shape[0:4] + (1,))
            save_im_data(kl_map, filename + '_kl')
            #y_pred, predicted_sigma = tf.split(y_pred, 2, -1)
            # Take the first n channels from y_pred, which correspond to the mean image prediction.
            y_pred = y_pred[:, :, :, :, :y_true.shape[-1]]
            if self._multi_image_normalisation:
                y_true = y_true / (np.mean(y_true[:, :, :, :, self._se_idx-1:self._se_idx+2], -1, keepdims=True) + 1e-3)
                y_pred = y_pred / (np.mean(y_pred[:, :, :, :, self._se_idx-1:self._se_idx+2], -1, keepdims=True) + 1e-3)
            else:
                y_true = y_true / (np.mean(y_true[:, :, :, :, self._se_idx:self._se_idx+1], -1, keepdims=True) + 1e-3)
                y_pred = y_pred / (np.mean(y_pred[:, :, :, :, self._se_idx:self._se_idx+1], -1, keepdims=True) + 1e-3)

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
