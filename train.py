#!/usr/bin/env python3


from signals import SignalGenerationLayer
import tensorflow_probability as tfp

import os
import numpy as np
import argparse
import configparser
import math
from model import EncoderTrainer
import tensorflow as tf
from tensorflow import keras


def fine_tune_loss_fn(y_true, y_pred, student_t_df=None, sigma=0.08):
    """
    The std_dev of 0.08 is estimated from real data
    """
    mask = y_true[:, :, :, :, -1:]
    sigma = tf.reduce_mean(y_pred[:, :, :, :, -1:])
    y_pred = y_pred[:, :, :, :, :-1]
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
        nll = -(-tf.math.log(sigma) - np.log(np.sqrt(2.0 * np.pi)) - 0.5 * tf.square(residual / sigma))

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


def prepare_dataset(real_data, model, crop_size=20):
    # Prepare the real data
    real_data = np.float32(real_data[:, 10:-10, 10:-10, :, :])
    # Mask the data and make some predictions to provide a prior distribution
    predicted_distribution, _ = model.predict(real_data[:, :, :, :, :-1] * real_data[:, :, :, :, -1:])
    predicted_distribution = predicted_distribution[:, :, :, :, 0:4]

    real_dataset = tf.data.Dataset.from_tensor_slices((real_data, predicted_distribution))

    def map_func2(data, predicted_distribution):
        data_shape = data.shape.as_list()
        new_shape = data_shape[0:2] + [-1, ]
        data = tf.reshape(data, new_shape)

        predicted_distribution_shape = predicted_distribution.shape.as_list()
        predicted_distribution = tf.reshape(predicted_distribution, new_shape)

        # concatenate to crop
        crop_data = tf.concat([data, predicted_distribution], -1)
        crop_data = tf.image.random_crop(value=crop_data, size=(crop_size, crop_size, crop_data.shape[-1]))

        # Separate out again
        predicted_distribution = crop_data[:, :, -predicted_distribution.shape.as_list()[-1]:]
        predicted_distribution = tf.reshape(predicted_distribution, [crop_size, crop_size] + predicted_distribution_shape[-2:])

        data = crop_data[:, :, :data.shape[-1]]
        data = tf.reshape(data, [crop_size, crop_size] + data_shape[-2:])
        mask = data[:, :, :, -1:]

        data = data[:, :, :, :-1] * data[:, :, :, -1:]
        # concat the mask
        data = tf.concat([data, mask], -1)

        predicted_distribution = tf.concat([predicted_distribution, mask], -1)

        return (data[:, :, :, :-1], mask), {'predictions': predicted_distribution, 'predicted_images': data}

    real_dataset = real_dataset.map(map_func2)
    real_dataset = real_dataset.shuffle(10000)
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
    log_stds = transform_std(log_stds)

    # Extract the OEF and DBV and transform them
    predictions = tf.concat([predictions[:, :, :, :, 0:1], predictions[:, :, :, :, 2:3]], -1)

    predictions = forward_transform(predictions)

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


if __name__ == '__main__':
    import wandb
    from wandb.keras import WandbCallback

    wandb.init(project='qbold_inference', entity='ivorsimpson')
    wb_config = wandb.config

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
    kl_weight = 1.0
    smoothness_weight = 1.0
    # Switching to None will use a Gaussian error distribution
    student_t_df = 10
    dropout_rate = 0.0
    use_layer_norm = False
    use_system_constants = False
    no_pt_epochs = 5
    no_ft_epochs = 50
    pt_lr = 1e-3
    ft_lr = 1e-3
    im_loss_sigma = 0.08
    no_intermediate_layers = 1
    predict_hct = False
    crop_size = 32

    data_file = np.load(args.f)
    x = data_file['x']
    y = data_file['y']

    train_conv = True
    # If we're building a convolutional model, reshape the synthetic data to look like images, note we only do
    # 1x1x1 convs for pre-training
    if train_conv:
        # Reshape to being more image like for layer normalisation (if we use this)
        x = np.reshape(x, (-1, 10, 10, 5, 11))
        y = np.reshape(y, (-1, 10, 10, 5, 3))

    # Separate into training/testing data
    # Keep 10% for validation
    no_examples = x.shape[0]
    no_valid_examples = no_examples // 10
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

    trainer = EncoderTrainer(no_units=no_units, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate, no_intermediate_layers=no_intermediate_layers)
    model = trainer.create_encoder()

    def synth_loss(x, y):
        return trainer.synthetic_data_loss(x, y)

    def oef_metric(x, y):
        return trainer.oef_metric(x, y)

    def dbv_metric(x, y):
        return trainer.dbv_metric(x, y)

    def r2p_metric(x, y):
        return trainer.r2p_metric(x, y)

    model.compile(optimiser, loss=[synth_loss, None],
                  metrics=[[oef_metric, dbv_metric, r2p_metric], None])

    model.fit(synthetic_dataset, epochs=no_pt_epochs, validation_data=(valid_x, valid_y))

    # Load real data for fine-tuning
    ase_data = np.load(f'{args.d}/ASE_scan.npy')
    ase_inf_data = np.load(f'{args.d}/ASE_INF.npy')
    ase_sup_data = np.load(f'{args.d}/ASE_SUP.npy')

    train_data = np.concatenate([ase_data, ase_inf_data, ase_sup_data], axis=0)
    train_dataset = prepare_dataset(train_data, model, crop_size)

    hyperv_data = np.load(f'{args.d}/hyperv_ase.npy')
    baseline_data = np.load(f'{args.d}/baseline_ase.npy')

    study_data = np.concatenate([hyperv_data, baseline_data], axis=0)
    study_dataset = prepare_dataset(study_data, model, crop_size)

    if not os.path.exists('pt'):
        os.makedirs('pt')

    model.save('pt/model.h5')

    save_predictions(model, baseline_data, 'pt/baseline')
    save_predictions(model, hyperv_data, 'pt/hyperv')

    full_optimiser = tf.keras.optimizers.Adam(learning_rate=ft_lr)
    input_3d = keras.layers.Input((crop_size, crop_size, 8, 11))
    input_mask = keras.layers.Input((crop_size, crop_size, 8, 1))
    net = input_3d
    _, predicted_distribution = model(net)

    sampled_oef_dbv = ReparamTrickLayer()((predicted_distribution, input_mask))

    sigma_layer = tfp.layers.VariableLayer(shape=(1,), dtype=tf.dtypes.float32, activation=tf.exp,
                                           initializer=tf.keras.initializers.constant(np.log(im_loss_sigma)))

    params['simulate_noise'] = 'False'
    output = SignalGenerationLayer(params, True, True)(sampled_oef_dbv)
    output = tf.concat([output, tf.ones_like(output[:, :, :, :, 0:1]) * sigma_layer(output)], -1)
    full_model = keras.Model(inputs=[input_3d, input_mask],
                             outputs={'predictions': predicted_distribution, 'predicted_images': output})


    def predictions_loss(t, p):
        return kl_loss(t, p) * kl_weight + smoothness_loss(t, p) * smoothness_weight

    def sigma_metric(t, p):
        return tf.reduce_mean(p[:, :, :, :, -1:])

    full_model.compile(full_optimiser,
                       loss={'predicted_images': lambda _x, _y: fine_tune_loss_fn(_x, _y, student_t_df=student_t_df,
                                                                                  sigma=im_loss_sigma),
                             'predictions': predictions_loss},
                       metrics={'predictions': [smoothness_loss, kl_loss], 'predicted_images': sigma_metric})


    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.1


    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    full_model.fit(train_dataset, validation_data=study_dataset, steps_per_epoch=100, epochs=no_ft_epochs,
                   validation_steps=1, callbacks=[scheduler_callback, WandbCallback()])

    model.save('model.h5')

    save_predictions(model, baseline_data, 'baseline', use_first_op=False)
    save_predictions(model, hyperv_data, 'hyperv', use_first_op=False)
