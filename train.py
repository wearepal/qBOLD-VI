#!/usr/bin/env python3


from signals import SignalGenerationLayer, create_synthetic_dataset

import os
import numpy as np
import argparse
import configparser
from model import EncoderTrainer
import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback


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


def prepare_dataset(real_data, model, crop_size=20, training=True):
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
        predicted_distribution = tf.reshape(predicted_distribution,
                                            [crop_size, crop_size] + predicted_distribution_shape[-2:])

        data = crop_data[:, :, :data.shape[-1]]
        data = tf.reshape(data, [crop_size, crop_size] + data_shape[-2:])
        mask = data[:, :, :, -1:]

        data = data[:, :, :, :-1] * data[:, :, :, -1:]
        # concat the mask
        data = tf.concat([data, mask], -1)

        predicted_distribution = tf.concat([predicted_distribution, mask], -1)

        return (data[:, :, :, :-1], mask), {'predictions': predicted_distribution, 'predicted_images': data}

    real_dataset = real_dataset.map(map_func2)
    if training:
        real_dataset = real_dataset.shuffle(10000)
        real_dataset = real_dataset.batch(6, drop_remainder=True)
    else:
        real_dataset = real_dataset.batch(12, drop_remainder=True)

    real_dataset = real_dataset.repeat(-1)
    return real_dataset


def load_synthetic_dataset(filename):
    data_file = np.load(filename)
    x = data_file['x']
    y = data_file['y']
    return x, y


def prepare_synthetic_dataset(x, y):
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
    return synthetic_dataset, (valid_x, valid_y)


def setup_argparser(defaults_dict):
    parser = argparse.ArgumentParser(description='Train neural network for parameter estimation')

    parser.add_argument('-f', default='synthetic_data.npz', help='path to synthetic data file')
    parser.add_argument('-d', default='/home/data/qbold/', help='path to the real data directory')
    parser.add_argument('--no_units', type=int, default=defaults_dict['no_units'])
    parser.add_argument('--no_pt_epochs', type=int, default=defaults_dict['no_pt_epochs'])
    parser.add_argument('--no_ft_epochs', type=int, default=defaults_dict['no_ft_epochs'])
    parser.add_argument('--student_t_df', type=int, default=defaults_dict['student_t_df'])
    parser.add_argument('--crop_size', type=int, default=defaults_dict['crop_size'])
    parser.add_argument('--no_intermediate_layers', type=int, default=defaults_dict['no_intermediate_layers'])
    parser.add_argument('--kl_weight', type=float, default=defaults_dict['kl_weight'])
    parser.add_argument('--smoothness_weight', type=float, default=defaults_dict['smoothness_weight'])
    parser.add_argument('--pt_lr', type=float, default=defaults_dict['pt_lr'])
    parser.add_argument('--ft_lr', type=float, default=defaults_dict['ft_lr'])
    parser.add_argument('--dropout_rate', type=float, default=defaults_dict['dropout_rate'])
    parser.add_argument('--im_loss_sigma', type=float, default=defaults_dict['im_loss_sigma'])
    parser.add_argument('--use_layer_norm', type=bool, default=defaults_dict['use_layer_norm'])
    parser.add_argument('--use_r2p_loss', type=bool, default=defaults_dict['use_r2p_loss'])
    parser.add_argument('--multi_image_normalisation', type=bool, default=defaults_dict['multi_image_normalisation'])
    parser.add_argument('--activation', default=defaults_dict['activation'])
    parser.add_argument('--misalign_prob', type=float, default=defaults_dict['misalign_prob'])
    parser.add_argument('--use_blood', type=bool, default=defaults_dict['use_blood'])
    parser.add_argument('--full_model', type=bool, default=defaults_dict['full_model'])
    parser.add_argument('--save_directory', default=None)
    parser.add_argument('--use_population_prior', type=bool, default=defaults_dict['use_population_prior'])
    parser.add_argument('--use_wandb', type=bool, default=defaults_dict['use_wandb'])

    return parser


def get_defaults():
    defaults = dict(
        no_units=30,
        no_intermediate_layers=1,
        student_t_df=2,  # Switching to None will use a Gaussian error distribution
        pt_lr=5e-5,
        ft_lr=5e-3,
        kl_weight=1.0,
        smoothness_weight=1.0,
        dropout_rate=0.0,
        no_pt_epochs=5,
        no_ft_epochs=40,
        im_loss_sigma=0.08,
        crop_size=16,
        use_layer_norm=False,
        activation='relu',
        use_r2p_loss=False,
        multi_image_normalisation=True,
        full_model=True,
        use_blood=True,
        misalign_prob=0.2,
        use_population_prior=False,
        use_wandb=True
    )
    return defaults


def train_model(config_dict):
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']
    optimiser = tf.keras.optimizers.Adam(learning_rate=config_dict.pt_lr)

    """if wb_config.use_system_constants:
        system_constants = get_constants(params)
    else:
        system_constants = None"""

    trainer = EncoderTrainer(system_params=params,
                             no_units=config_dict.no_units,
                             use_layer_norm=config_dict.use_layer_norm,
                             dropout_rate=config_dict.dropout_rate,
                             no_intermediate_layers=config_dict.no_intermediate_layers,
                             student_t_df=config_dict.student_t_df,
                             initial_im_sigma=config_dict.im_loss_sigma,
                             activation_type=config_dict.activation,
                             multi_image_normalisation=config_dict.multi_image_normalisation,
                             )

    model = trainer.create_encoder()

    if not config_dict.use_population_prior:
        def synth_loss(x, y):
            return trainer.synthetic_data_loss(x, y, config_dict.use_r2p_loss)

        def oef_metric(x, y):
            return trainer.oef_metric(x, y)

        def dbv_metric(x, y):
            return trainer.dbv_metric(x, y)

        def r2p_metric(x, y):
            return trainer.r2p_metric(x, y)

        model.compile(optimiser, loss=[synth_loss, None],
                      metrics=[[oef_metric, dbv_metric, r2p_metric], None])

        # x, y = load_synthetic_dataset(args.f)
        x, y = create_synthetic_dataset(params, config_dict.full_model, config_dict.use_blood,
                                        config_dict.misalign_prob)
        synthetic_dataset, synthetic_validation = prepare_synthetic_dataset(x, y)
        model.fit(synthetic_dataset, epochs=config_dict.no_pt_epochs, validation_data=synthetic_validation,
                  callbacks=[tf.keras.callbacks.TerminateOnNaN()])

    if not os.path.exists(config_dict.d):
        raise Exception('Real data directory not found')

    # Load real data for fine-tuning, using the model trained on synthetic data for priors
    ase_data = np.load(f'{config_dict.d}/ASE_scan.npy')
    ase_inf_data = np.load(f'{config_dict.d}/ASE_INF.npy')
    ase_sup_data = np.load(f'{config_dict.d}/ASE_SUP.npy')

    train_data = np.concatenate([ase_data, ase_inf_data, ase_sup_data], axis=0)
    train_dataset = prepare_dataset(train_data, model, config_dict.crop_size)

    hyperv_data = np.load(f'{config_dict.d}/hyperv_ase.npy')
    baseline_data = np.load(f'{config_dict.d}/baseline_ase.npy')

    study_data = np.concatenate([hyperv_data, baseline_data], axis=0)
    study_dataset = prepare_dataset(study_data, model, 76, training=False)

    # If we're not using the population prior we may want to save predictions from our initial model
    if not config_dict.use_population_prior:
        trainer.estimate_population_param_distribution(model, baseline_data)

        if config_dict.save_directory is not None:
            if not os.path.exists(config_dict.save_directory):
                os.makedirs(config_dict.save_directory)
            model.save_weights(config_dict.save_directory + '/pt_model.h5')
            trainer.save_predictions(model, baseline_data, config_dict.save_directory + '/pt_baseline')
            trainer.save_predictions(model, hyperv_data, config_dict.save_directory + '/pt_hyperv')

    full_optimiser = tf.keras.optimizers.Adam(learning_rate=config_dict.ft_lr)

    input_3d = keras.layers.Input((None, None, 8, 11))
    input_mask = keras.layers.Input((None, None, 8, 1))
    params['simulate_noise'] = 'False'
    sig_gen_layer = SignalGenerationLayer(params, config_dict.full_model, config_dict.use_blood)
    full_model = trainer.build_fine_tuner(model, sig_gen_layer, input_3d, input_mask,
                                          population_prior=config_dict.use_population_prior)

    def fine_tune_loss(x, y):
        return trainer.fine_tune_loss_fn(x, y)

    def predictions_loss(t, p):
        return EncoderTrainer.kl_loss(t, p, config_dict.use_population_prior) * config_dict.kl_weight + \
               EncoderTrainer.smoothness_loss(t, p) * config_dict.smoothness_weight

    def sigma_metric(t, p):
        return tf.reduce_mean(p[:, :, :, :, -1:])

    class ELBOCallback(tf.keras.callbacks.Callback):
        def __init__(self, dataset):
            self._iter = iter(dataset)

        def on_epoch_end(self, epoch, logs=None):
            data, y = next(self._iter)
            nll = 0.0
            for i in range(10):
                predictions = self.model.predict(data)
                nll += fine_tune_loss(y['predicted_images'], predictions['predicted_images'])
            nll = nll / 10.0

            kl = EncoderTrainer.kl_loss(y['predictions'], predictions['predictions'], config_dict.use_population_prior)
            smoothness = EncoderTrainer.smoothness_loss(y['predictions'], predictions['predictions'])
            metrics = {'val_nll': nll, 'val_elbo': nll + kl, 'val_elbo_smooth': nll + kl + smoothness,
                       'val_smoothness': smoothness, 'val_kl': kl}

            wandb.log(metrics)

    elbo_callback = ELBOCallback(study_dataset)

    full_model.compile(full_optimiser,
                       loss={'predicted_images': fine_tune_loss,
                             'predictions': predictions_loss},
                       metrics={'predictions': [EncoderTrainer.smoothness_loss, EncoderTrainer.kl_loss],
                                'predicted_images': sigma_metric})

    def scheduler(epoch, lr):
        if epoch < 20:
            return lr
        else:
            return lr * 0.8

    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    full_model.fit(train_dataset, steps_per_epoch=100, epochs=config_dict.no_ft_epochs,
                   callbacks=[scheduler_callback, WandbCallback(), elbo_callback,
                              tf.keras.callbacks.TerminateOnNaN()])
    trainer.estimate_population_param_distribution(model, baseline_data)

    if config_dict.save_directory is not None:
        if not os.path.exists(config_dict.save_directory):
            os.makedirs(config_dict.save_directory)
        model.save_weights(config_dict.save_directory + '/final_model.h5')
        trainer.save_predictions(model, baseline_data, config_dict.save_directory + '/baseline', use_first_op=False)
        trainer.save_predictions(model, hyperv_data, config_dict.save_directory + '/hyperv', use_first_op=False)


if __name__ == '__main__':
    import sys
    import yaml

    yaml_file = None
    # If we have a single argument and it's a yaml file, read the config from there
    if (len(sys.argv) == 2) and (".yaml" in sys.argv[1]):
        # Read the yaml filename
        yaml_file = sys.argv[1]
        # Remove it from the input arguments to also allow the default argparser
        sys.argv = [sys.argv[0]]

    cmd_parser = setup_argparser(get_defaults())
    args = cmd_parser.parse_args()
    args = vars(args)

    if yaml_file is not None:
        opt = yaml.load(open(yaml_file), Loader=yaml.FullLoader)
        # Overwrite defaults with yaml config, making sure we use the correct types
        for key, val in opt.items():
            if args.get(key):
                args[key] = type(args.get(key))(val)
            else:
                args[key] = val

    if args['use_wandb']:
        wandb.init(project='qbold_inference', entity='ivorsimpson')
        if not args.get('name') is None:
            wandb.run.name = args['name']

        wandb.config.update(args)
        train_model(wandb.config)

    else:
        train_model(args)
