#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from signals import SignalGenerationLayer

import os
import numpy as np
import argparse
import configparser


def create_model(use_conv=True, system_constants=None, no_units=18, use_layer_norm=False, dropout_rate=0.0,
                 no_intermediate_layers=1):
    """
    @param: use_conv (Boolean) : whether to use a convolution (1x1x1) or MLP model
    @params: system_constants (array): If not None, perform a dense transformation and multiply with first level representation
    @params: no_units (unsigned int): The number of units for each level of the network
    @params: use_layer_norm (Boolean) : Perform layer normalisation
    @params: dropout_rate (float, 0-1) : perform dropout
    @params: no_intermediate layers (unsigned int) : the number of extra layers apart from the first and last
    """
    def create_layer(_no_units, activation='gelu'):
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
        if dropout_rate > 0.0:
            _net = keras.layers.Dropout(dropout_rate)(_net)
        if use_layer_norm:
            _net = tfa.layers.GroupNormalization(groups=1, axis=-1)(_net)
        return _net

    # Create the initial layer
    net = create_layer(no_units)(net)
    net = add_normalizer(net)

    # If we passed in the system constants, we can apply a dense layer to then multiply them with the network
    if system_constants is not None:
        const_net = keras.layers.Dense(no_units)(system_constants)
        net = keras.layers.Multiply()([net, keras.layers.Reshape((1, 1, 1, -1))(const_net)])

    # Add intermediate layers layers
    for i in range(no_intermediate_layers):
        net = add_normalizer(net)
        net = create_layer(no_units)(net)

    net = add_normalizer(net)
    # Create the penultimate layer, leaving net available for more processing
    net_penultimate = create_layer(no_units)(net)

    if use_conv:
        # Add a second output that uses 3x3x3 convs
        ki = tf.keras.initializers.TruncatedNormal(stddev=0.05)  # GlorotNormal()
        second_net = keras.layers.Conv3D(no_units, kernel_size=(3, 3, 1), activation='gelu', padding='same',
                                         kernel_initializer=ki)(net)
        second_net = add_normalizer(second_net)
        second_net = keras.layers.Conv3D(no_units, kernel_size=(3, 3, 1), activation='gelu', padding='same',
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


def forward_transform(logit):
    # Define the forward transform of the predicted parameters to OEF/DBV
    oef, dbv = tf.split(logit, 2, -1)
    oef = tf.nn.sigmoid(oef) * 0.8
    dbv = tf.nn.sigmoid(dbv) * 0.3
    output = tf.concat([oef, dbv], axis=-1)
    return output


def logit(signal):
    # Inverse sigmoid function
    return tf.math.log(signal / (1.0 - signal))


def backwards_transform(signal):
    # Define how to backwards transform OEF/DBV to the same parameterisation used by the NN
    oef, dbv = tf.split(signal, 2, -1)
    oef = logit(oef / 0.8)
    dbv = logit(dbv / 0.3)
    output = tf.concat([oef, dbv], axis=-1)
    return output


def loss_fn(y_true, y_pred):
    # Reshape the data such that we can work with either volumes or single voxels
    y_true = tf.reshape(y_true, (-1, 2))
    # Backwards transform the true values (so we can define our distribution in the parameter space)
    y_true = backwards_transform(y_true)
    y_pred = tf.reshape(y_pred, (-1, 4))

    oef_mean = y_pred[:, 0]
    oef_log_std = y_pred[:, 1]
    dbv_mean = y_pred[:, 2]
    dbv_log_std = y_pred[:, 3]
    # Gaussian negative log likelihoods
    oef_nll = -(-oef_log_std - (1.0 / 2.0) * ((y_true[:, 0] - oef_mean) / tf.exp(oef_log_std)) ** 2)
    dbv_nll = -(-dbv_log_std - (1.0 / 2.0) * ((y_true[:, 1] - dbv_mean) / tf.exp(dbv_log_std)) ** 2)

    nll = tf.add(oef_nll, dbv_nll)

    return tf.reduce_mean(nll)


def oef_dbv_metrics(y_true, y_pred, oef):
    """
    Produce the MSE of the predictions of OEF or DBV
    @param oef is a boolean, if False produces the output for DBV
    """
    # Reshape the data such that we can work with either volumes or single voxels
    y_true = tf.reshape(y_true, (-1, 2))
    y_pred = tf.reshape(y_pred, (-1, 4))
    # keras.backend.print_tensor(tf.reduce_mean(tf.exp(y_pred[:,1])))
    means = tf.stack([y_pred[:, 0], y_pred[:, 2]], -1)
    means = forward_transform(means)
    residual = means - y_true
    if oef:
        residual = residual[:, 0]
    else:
        residual = residual[:, 1]
    return tf.reduce_mean(tf.square(residual))


def oef_metric(y_true, y_pred):
    return oef_dbv_metrics(y_true, y_pred, True)


def dbv_metric(y_true, y_pred):
    return oef_dbv_metrics(y_true, y_pred, False)


class ReparamTrickLayer(keras.layers.Layer):
    # Draw samples of OEF and DBV from the predicted distributions
    def call(self, input, *args, **kwargs):
        oef_sample = input[:, :, :, :, 0] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
            input[:, :, :, :, 1])
        dbv_sample = input[:, :, :, :, 2] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
            input[:, :, :, :, 3])

        samples = tf.stack([oef_sample, dbv_sample], -1)
        # Forward transform
        samples = forward_transform(samples)
        # Clip to avoid really tiny/large values breaking the forward model with nans
        samples = tf.clip_by_value(samples, clip_value_min=1e-3, clip_value_max=0.99)
        return samples


def fine_tune_loss_fn(y_true, y_pred, student_t_df=None, sigma=0.08):
    """
    The std_dev of 0.08 is estimated from real data
    """
    import tensorflow_probability as tfp
    mask = y_true[:, :, :, :, -1:]
    # Normalise and mask the predictions/real data
    y_true = y_true / (tf.reduce_mean(y_true[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
    y_pred = y_pred / (tf.reduce_mean(y_pred[:, :, :, :, 1:4], -1, keepdims=True) + 1e-3)
    y_true = tf.where(mask > 0, tf.math.log(y_true), tf.zeros_like(y_true))
    y_pred = tf.where(mask > 0, tf.math.log(y_pred), tf.zeros_like(y_pred))

    # Calculate the residual difference between our normalised data
    residual = y_true[:, :, :, :, :-1] - y_pred
    residual = tf.reshape(residual, (-1, 11))
    mask = tf.reshape(mask, (-1, 1))

    # Optionally use a student-t distribution (with heavy tails) or a Gaussian distribution
    if student_t_df is not None:
        dist = tfp.distributions.StudentT(df=student_t_df, loc=0.0, scale=sigma)
        nll = -dist.log_prob(residual)
    else:
        nll = -(-sigma - np.sqrt(2.0 * np.pi) - (1.0 / 2.0) * (residual / sigma) ** 2)

    nll = tf.reduce_sum(nll, -1, keepdims=True)
    nll = nll * mask

    return tf.reduce_sum(nll) / tf.reduce_sum(mask)


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


def get_constants(params):
    # Put the system constants into an array
    dchi = float(params['dchi'])
    hct = float(params['hct'])
    te = float(params['te'])
    r2t = float(params['r2t'])
    tr = float(params['tr'])
    ti = float(params['ti'])
    t1b = float(params['t1b'])
    consts = np.array([dchi, hct, te, r2t, tr, ti, t1b], dtype=np.float32)
    taus = tf.range(float(params['tau_start']), float(params['tau_end']),
                    float(params['tau_step']), dtype=tf.float32)
    consts = tf.concat([consts, taus], 0)
    consts = tf.reshape(consts, (1, -1))
    return consts


def smoothness_loss(true_params, pred_params):
    # Define a total variation smoothness term for the predicted means
    q_oef_mean, q_oef_log_std, q_dbv_mean, q_dbv_log_std = tf.split(pred_params, 4, -1)
    pred_params = tf.concat([q_oef_mean, q_dbv_mean], -1)

    diff_x = pred_params[:, :-1, :, :, :] - pred_params[:, 1:, :, :, :]
    diff_y = pred_params[:, :, :-1, :, :] - pred_params[:, :, 1:, :, :]
    diff_z = pred_params[:, :, :, :-1, :] - pred_params[:, :, :, 1:, :]

    diffs = tf.reduce_mean(tf.abs(diff_x)) + tf.reduce_mean(tf.abs(diff_y)) + tf.reduce_mean(tf.abs(diff_z))
    return diffs


def prepare_dataset(real_data, model):
    # Prepare the real data
    real_data = np.float32(real_data)
    # Mask the data and make some predictions to provide a prior distribution
    predicted_distribution, _ = model.predict(real_data[:, :, :, :, :-1] * real_data[:, :, :, :, -1:])

    real_dataset = tf.data.Dataset.from_tensor_slices((real_data, predicted_distribution))

    def map_func2(data, predicted_distribution):
        data_shape = data.shape.as_list()
        new_shape = data_shape[0:2] + [-1, ]
        data = tf.reshape(data, new_shape)

        predicted_distribution_shape = predicted_distribution.shape.as_list()
        predicted_distribution = tf.reshape(predicted_distribution, new_shape)

        # concatenate to crop
        crop_data = tf.concat([data, predicted_distribution], -1)
        crop_data = tf.image.random_crop(value=crop_data, size=(20, 20, crop_data.shape[-1]))

        # Separate out again
        predicted_distribution = crop_data[:, :, -predicted_distribution.shape.as_list()[-1]:]
        predicted_distribution = tf.reshape(predicted_distribution, [20, 20] + predicted_distribution_shape[-2:])

        data = crop_data[:, :, :data.shape[-1]]
        data = tf.reshape(data, [20, 20] + data_shape[-2:])
        mask = data[:, :, :, -1:]

        data = data[:, :, :, :-1] * data[:, :, :, -1:]
        # concat the mask
        data = tf.concat([data, mask], -1)

        predicted_distribution = tf.concat([predicted_distribution, mask], -1)

        return data[:, :, :, :-1], {'predictions': predicted_distribution, 'predicted_images': data}

    real_dataset = real_dataset.map(map_func2)
    real_dataset = real_dataset.batch(6, drop_remainder=True)
    real_dataset = real_dataset.repeat(-1)
    return real_dataset


def save_predictions(model, data, filename, use_first_op=True):
    import nibabel as nib

    predictions, predictions2 = model.predict(data[:, :, :, :, :-1] * data[:, :, :, :, -1:])
    if use_first_op is False:
        predictions = predictions2

    # Get the log stds, but don't transform them. Their meaning is complicated because of the forward transformation
    log_stds = tf.concat([predictions[:, :, :, :, 1:2], predictions[:, :, :, :, 3:4]], -1)

    # Extract the OEF and DBV and transform them
    predictions = tf.concat([predictions[:, :, :, :, 0:1], predictions[:, :, :, :, 2:3]], -1)
    predictions = forward_transform(predictions)



    def save_im_data(im_data, _filename):
        images = np.split(im_data, data.shape[0], axis=0)
        images = np.squeeze(np.concatenate(images, axis=-1), 0)
        affine = np.eye(4)
        array_img = nib.Nifti1Image(images, affine)
        nib.save(array_img, _filename+'.nii.gz')

    save_im_data(predictions[:, :, :, :, 0:1], filename + '_oef')
    save_im_data(predictions[:, :, :, :, 1:2], filename + '_dbv')
    save_im_data(log_stds, filename + '_logstds')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']
    parser = argparse.ArgumentParser(description='Train neural network for parameter estimation')

    parser.add_argument('-f', default='synthetic_data.npz', help='path to synthetic data file')
    parser.add_argument('-d', default='', help='path to the real data directory')

    args = parser.parse_args()

    if not os.path.exists(args.d):
        raise Exception('Real data directory not found')


    no_units = 20
    kl_weight = 0.5
    smoothness_weight = 0.5
    # Switching to None will use a Gaussian error distribution
    student_t_df = None
    dropout_rate = 0.0
    use_layer_norm = False
    use_system_constants = False
    no_pt_epochs = 50
    no_ft_epochs = 50
    pt_lr = 1e-3
    ft_lr = 1e-3
    im_loss_sigma = 0.08
    no_intermediate_layers = 1

    data_file = np.load(args.f)
    x = data_file['x']
    y = data_file['y']

    train_conv = True
    # If we're building a convolutional model, reshape the synthetic data to look like images, note we only do
    # 1x1x1 convs for pre-training
    if train_conv:
        # Reshape to being more image like for layer normalisation (if we use this)
        x = np.reshape(x, (-1, 10, 10, 5, 11))
        y = np.reshape(y, (-1, 10, 10, 5, 2))

    # Separate into training/testing data
    # Keep 10% for validation
    no_examples = x.shape[0]
    no_valid_examples = no_examples//10
    train_x = x[:-no_valid_examples, ...]
    train_y = y[:-no_valid_examples, ...]
    valid_x = x[-no_valid_examples:, ...]
    valid_y = y[-no_valid_examples:, ...]

    synthetic_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

    synthetic_dataset = synthetic_dataset.shuffle(10000)
    synthetic_dataset = synthetic_dataset.batch(6)

    optimiser = tf.keras.optimizers.Adam(learning_rate=pt_lr)

    if use_system_constants:
        system_constants = get_constants(params)
    else:
        system_constants = None

    model = create_model(use_conv=train_conv, no_units=no_units, use_layer_norm=use_layer_norm,
                         dropout_rate=dropout_rate, system_constants=system_constants,
                         no_intermediate_layers=no_intermediate_layers)

    model.compile(optimiser, loss=[loss_fn, None], metrics=[[oef_metric, dbv_metric], None])

    mc = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1)

    model.fit(synthetic_dataset, epochs=no_pt_epochs, callbacks=[mc], validation_data=(valid_x, valid_y))

    # Load real data for fine-tuning
    real_data = np.load(f'{args.d}/hyperv_ase.npy')
    real_dataset = prepare_dataset(real_data, model)

    valid_data = np.load(f'{args.d}/baseline_ase.npy')

    valid_dataset = prepare_dataset(valid_data, model)

    combined_data = np.concatenate([real_data, valid_data], axis=0)
    combined_dataset = prepare_dataset(combined_data, model)

    if not os.path.exists('pt'):
        os.makedirs('pt')

    save_predictions(model, valid_data, 'pt/baseline')
    save_predictions(model, real_data, 'pt/hyperv')

    full_optimiser = tf.keras.optimizers.Adam(learning_rate=ft_lr)
    input_3d = keras.layers.Input((20, 20, 8, 11))
    net = input_3d
    _, predicted_distribution = model(net)

    sampled_oef_dbv = ReparamTrickLayer()(predicted_distribution)

    params['simulate_noise'] = 'False'
    output = SignalGenerationLayer(params, False, True)(sampled_oef_dbv)
    full_model = keras.Model(inputs=[input_3d],
                             outputs={'predictions': predicted_distribution, 'predicted_images': output})

    def predictions_loss(t, p):
        return kl_loss(t, p) * kl_weight + smoothness_loss(t, p) * smoothness_weight


    full_model.compile(full_optimiser,
                       loss={'predicted_images': lambda _x, _y: fine_tune_loss_fn(_x, _y, student_t_df=student_t_df,
                                                                                  sigma=im_loss_sigma),
                             'predictions': predictions_loss},
                       metrics={'predictions': [smoothness_loss, kl_loss]})
    full_model.fit(combined_dataset, validation_data=combined_dataset, steps_per_epoch=100, epochs=no_ft_epochs,
                   validation_steps=1)
    save_predictions(model, valid_data, 'baseline', use_first_op=False)
    save_predictions(model, real_data, 'hyperv', use_first_op=False)
