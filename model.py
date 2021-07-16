# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Store the model code

import tensorflow as tf
from tensorflow import keras


class EncoderTrainer:
    def __init__(self,
                 no_intermediate_layers=1,
                 no_units=10,
                 use_layer_norm=False,
                 dropout_rate=0.0,
                 activation_type='gelu'
                 ):
        self._no_intermediate_layers = no_intermediate_layers
        self._no_units = no_units
        self._use_layer_norm = use_layer_norm
        self._dropout_rate = dropout_rate
        self._activation_type = activation_type

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
            # Normalise based on the mean of tau =0 and adjacent tau values to minimise the effects of noise
            _data = _data / tf.reduce_mean(_data[:, 1:4], -1, keepdims=True)
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
            second_net = keras.layers.Conv3D(self._no_units, kernel_size=(3, 3, 1), activation=self._activation_type, padding='same',
                                             kernel_initializer=ki)(net)
            second_net = add_normalizer(second_net)
            second_net = keras.layers.Conv3D(self._no_units, kernel_size=(3, 3, 1), activation=self._activation_type, padding='same',
                                             kernel_initializer=ki)(second_net)

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
