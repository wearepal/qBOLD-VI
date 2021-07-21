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
                 multi_image_normalisation=True
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

    def create_encoder(self, use_conv=True, system_constants=None):
        """
        @param: use_conv (Boolean) : whether to use a convolution (1x1x1) or MLP model
        @params: system_constants (array): If not None, perform a dense transformation and multiply with first level representation
        @params: no_units (unsigned int): The number of units for each level of the network
        @params: use_layer_norm (Boolean) : Perform layer normalisation
        @params: dropout_rate (float, 0-1) : perform dropout
        @params: no_intermediate layers (unsigned int) : the number of extra layers apart from the first and last
        """

        def create_layer(_no_units, activation=self._activation_type):
            if use_conv:
                return keras.layers.Conv3D(_no_units, kernel_size=(1, 1, 1), activation=activation)
            else:
                return keras.layers.Dense(_no_units, activation=activation)

        def normalise_data(_data):
            # Do the normalisation as part of the model
            orig_shape = tf.shape(_data)
            _data = tf.reshape(_data, (-1, 11))
            _data = tf.clip_by_value(_data, 1e-2, 1e8)
            if self._multi_image_normalisation:
                # Normalise based on the mean of tau =0 and adjacent tau values to minimise the effects of noise
                _data = _data / tf.reduce_mean(_data[:, 1:4], -1, keepdims=True)
            else:
                _data = _data / tf.reduce_mean(_data[:, 2:3], -1, keepdims=True)
            # Take the logarithm
            _data = tf.math.log(_data)
            _data = tf.reshape(_data, orig_shape)
            return _data

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

        # Create the initial layer
        net = create_layer(self._no_units)(net)
        net = add_normalizer(net)

        # If we passed in the system constants, we can apply a dense layer to then multiply them with the network
        if system_constants is not None:
            const_net = keras.layers.Dense(self._no_units)(system_constants)
            net = keras.layers.Multiply()([net, keras.layers.Reshape((1, 1, 1, -1))(const_net)])

        # Add intermediate layers layers
        for i in range(self._no_intermediate_layers):
            net = add_normalizer(net)
            net = create_layer(self._no_units)(net)

        net = add_normalizer(net)
        # Create the penultimate layer, leaving net available for more processing
        net_penultimate = create_layer(self._no_units)(net)

        if use_conv:
            # Add a second output that uses 3x3x3 convs
            ki = tf.keras.initializers.TruncatedNormal(stddev=0.05)  # GlorotNormal()
            second_net = keras.layers.Conv3D(self._no_units, kernel_size=(3, 3, 1), activation=self._activation_type,
                                             padding='same', kernel_initializer=ki)(net)
            second_net = add_normalizer(second_net)
            second_net = keras.layers.Conv3D(self._no_units, kernel_size=(3, 3, 1), activation=self._activation_type,
                                             padding='same', kernel_initializer=ki)(second_net)

            # Add this to the penultimate output from the 1x1x1 network
            second_net = keras.layers.Add()([second_net, net_penultimate])
        else:
            second_net = net_penultimate

        # Create the final layer, which produces a mean and variance for OEF and DBV
        final_layer = create_layer(4, activation=None)
        # Create an output that just looks at individual voxels
        output = final_layer(net_penultimate)
        # Create another output that also looks at neighbourhoods
        second_net = final_layer(second_net)

        # Create the model with two outputs, one with 3x3 convs for fine-tuning, and one without.
        return keras.Model(inputs=[input], outputs=[output, second_net])

    def build_fine_tuner(self, encoder_model, signal_generation_layer, input_im, input_mask):
        net = input_im
        _, predicted_distribution = encoder_model(net)

        sampled_oef_dbv = ReparamTrickLayer(self)((predicted_distribution, input_mask))

        sigma_layer = tfp.layers.VariableLayer(shape=(1,), dtype=tf.dtypes.float32, activation=tf.exp,
                                               initializer=tf.keras.initializers.constant(np.log(self._initial_im_sigma)))

        output = signal_generation_layer(sampled_oef_dbv)
        output = tf.concat([output, tf.ones_like(output[:, :, :, :, 0:1]) * sigma_layer(output)], -1)
        full_model = keras.Model(inputs=[input_im, input_mask],
                                 outputs={'predictions': predicted_distribution, 'predicted_images': output})
        return full_model

    def transform_std(self, pred_stds):
        # Transform the predicted std-dev to the correct range
        return tf.tanh(pred_stds) * 3.0

    def forward_transform(self, logits):
        # Define the forward transform of the predicted parameters to OEF/DBV
        oef, dbv = tf.split(logits, 2, -1)
        oef = tf.nn.sigmoid(oef) * 0.8 + 0.025
        dbv = tf.nn.sigmoid(dbv) * 0.3 + 0.002
        # hct = tf.nn.sigmoid(hct) * 0.02 + 0.34
        output = tf.concat([oef, dbv], axis=-1)
        return output

    def oef_dbv_metrics(self, y_true, y_pred, oef_dbv_r2p=0):
        """
        Produce the MSE of the predictions of OEF or DBV
        @param oef is a boolean, if False produces the output for DBV
        """
        # Reshape the data such that we can work with either volumes or single voxels
        y_true = tf.reshape(y_true, (-1, 3))
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
            r2p = self.calculate_r2p(means[:, 0], means[:,1])
            residual = y_true[:, 2] - r2p

        return tf.reduce_mean(tf.square(residual))

    def oef_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 0)

    def dbv_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 1)

    def r2p_metric(self, y_true, y_pred):
        return self.oef_dbv_metrics(y_true, y_pred, 2)

    def backwards_transform(self, signal):
        # Define how to backwards transform OEF/DBV to the same parameterisation used by the NN
        oef, dbv = tf.split(signal, 2, -1)
        oef = logit((oef - 0.025) / 0.8)
        dbv = logit((dbv - 0.001) / 0.3)
        # hct = logit((hct - 0.34) / 0.02)
        output = tf.concat([oef, dbv], axis=-1)
        return output

    def synthetic_data_loss(self, y_true_orig, y_pred_orig, use_r2p_loss):
        # Reshape the data such that we can work with either volumes or single voxels
        y_true_orig = tf.reshape(y_true_orig, (-1, 3))
        # Backwards transform the true values (so we can define our distribution in the parameter space)
        y_true = self.backwards_transform(y_true_orig[:, 0:2])
        y_pred = tf.reshape(y_pred_orig, (-1, 4))

        oef_mean = y_pred[:, 0]
        oef_log_std = self.transform_std(y_pred[:, 1])
        dbv_mean = y_pred[:, 2]
        dbv_log_std = self.transform_std(y_pred[:, 3])

        def gaussian_nll(obs, mean, log_std):
            return -(-log_std - (1.0 / 2.0) * ((obs - mean) / tf.exp(log_std)) ** 2)

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

    def fine_tune_loss_fn(self, y_true, y_pred):
        """
        The std_dev of 0.08 is estimated from real data
        """
        mask = y_true[:, :, :, :, -1:]
        sigma = tf.reduce_mean(y_pred[:, :, :, :, -1:])
        y_pred = y_pred[:, :, :, :, :-1]

        # Normalise and mask the predictions/real data
        if self._multi_image_normalisation:
            y_true = y_true / (tf.reduce_mean(y_true[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
            y_pred = y_pred / (tf.reduce_mean(y_pred[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
        else:
            y_true = y_true / (tf.reduce_mean(y_true[:, :, :, :, 2:3], -1, keepdims=True) + 1e-3)
            y_pred = y_pred / (tf.reduce_mean(y_pred[:, :, :, :, 2:3], -1, keepdims=True) + 1e-3)

        y_true = tf.where(mask > 0, tf.math.log(y_true), tf.zeros_like(y_true))
        y_pred = tf.where(mask > 0, tf.math.log(y_pred), tf.zeros_like(y_pred))

        # Calculate the residual difference between our normalised data
        residual = y_true[:, :, :, :, :-1] - y_pred
        residual = tf.reshape(residual, (-1, 11))
        mask = tf.reshape(mask, (-1, 1))

        # Optionally use a student-t distribution (with heavy tails) or a Gaussian distribution
        if self._student_t_df is not None:
            dist = tfp.distributions.StudentT(df=self._student_t_df, loc=0.0, scale=sigma)
            nll = -dist.log_prob(residual)
        else:
            nll = -(-tf.math.log(sigma) - np.log(np.sqrt(2.0 * np.pi)) - 0.5 * tf.square(residual / sigma))

        nll = tf.reduce_sum(nll, -1, keepdims=True)
        nll = nll * mask

        return tf.reduce_sum(nll) / tf.reduce_sum(mask)

    @staticmethod
    def kl_loss(true, predicted):
        # Calculate the kullback-leibler divergence between the posterior and prior distibutions
        q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std = tf.split(predicted, 4, -1)
        p_oef_mean, p_oef_log_std, p_dbv_mean, p_dbv_log_std, mask = tf.split(true, 5, -1)

        def kl(q_mean, q_log_std, p_mean, p_log_std):
            result = tf.exp(q_log_std * 2 - p_log_std * 2) + tf.square(p_mean - q_mean) * tf.exp(p_log_std * -2.0)
            result = result + p_log_std * 2 - q_log_std * 2 - 1.0
            return result * 0.5

        kl_oef = kl(q_oef_mean, q_oef_log_std, p_oef_mean, p_oef_log_std)
        kl_dbv = kl(q_dbv_mean, q_dbv_log_std, p_dbv_mean, p_dbv_log_std)
        # Mask the KL
        kl_op = (kl_oef + kl_dbv) * mask
        kl_op = tf.where(mask > 0, kl_op, tf.zeros_like(kl_op))
        # keras.backend.print_tensor(kl_op)
        return tf.reduce_sum(kl_op) / tf.reduce_sum(mask)

    @staticmethod
    def smoothness_loss(true_params, pred_params):
        # Define a total variation smoothness term for the predicted means
        q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std = tf.split(pred_params, 4, -1)
        pred_params = tf.concat([q_oef_mean, q_dbv_mean], -1)

        diff_x = pred_params[:, :-1, :, :, :] - pred_params[:, 1:, :, :, :]
        diff_y = pred_params[:, :, :-1, :, :] - pred_params[:, :, 1:, :, :]
        diff_z = pred_params[:, :, :, :-1, :] - pred_params[:, :, :, 1:, :]

        diffs = tf.reduce_mean(tf.abs(diff_x)) + tf.reduce_mean(tf.abs(diff_y)) + tf.reduce_mean(tf.abs(diff_z))
        return diffs

    def save_predictions(self, model, data, filename, use_first_op=True):
        import nibabel as nib

        predictions, predictions2 = model.predict(data[:, :, :, :, :-1] * data[:, :, :, :, -1:])
        if use_first_op is False:
            predictions = predictions2

        # Get the log stds, but don't transform them. Their meaning is complicated because of the forward transformation
        log_stds = tf.concat([predictions[:, :, :, :, 1:2], predictions[:, :, :, :, 3:4]], -1)
        log_stds = self.transform_std(log_stds)

        # Extract the OEF and DBV and transform them
        predictions = tf.concat([predictions[:, :, :, :, 0:1], predictions[:, :, :, :, 2:3]], -1)

        predictions = self.forward_transform(predictions)

        def save_im_data(im_data, _filename):
            images = np.split(im_data, data.shape[0], axis=0)
            images = np.squeeze(np.concatenate(images, axis=-1), 0)
            affine = np.eye(4)
            array_img = nib.Nifti1Image(images, affine)
            nib.save(array_img, _filename + '.nii.gz')

        save_im_data(predictions[:, :, :, :, 0:1], filename + '_oef')
        save_im_data(predictions[:, :, :, :, 1:2], filename + '_dbv')
        # save_im_data(predictions[:, :, :, :, 2:3], filename + '_hct')
        save_im_data(log_stds, filename + '_logstds')
