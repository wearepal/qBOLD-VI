#!/usr/bin/env python3


from signals import SignalGenerationLayer, create_synthetic_dataset

import os
import numpy as np
import argparse
import configparser
from model import EncoderTrainer
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
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
    predicted_distribution, _, _ = model.predict(real_data[:, :, :, :, :-1] * real_data[:, :, :, :, -1:])

    if tf.shape(predicted_distribution)[-1] == 5:
        predicted_distribution = predicted_distribution[:, :, :, :, 0:5]
    else:
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
        real_dataset = real_dataset.batch(38, drop_remainder=True)
    else:
        real_dataset = real_dataset.batch(3, drop_remainder=True)

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
    synthetic_dataset = synthetic_dataset.batch(128)
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
    parser.add_argument('--channelwise_gating', type=bool, default=defaults_dict['channelwise_gating'])
    parser.add_argument('--full_model', type=bool, default=defaults_dict['full_model'])
    parser.add_argument('--save_directory', default=None)
    parser.add_argument('--use_population_prior', type=bool, default=defaults_dict['use_population_prior'])
    parser.add_argument('--inv_gamma_alpha', type=float, default=defaults_dict['inv_gamma_alpha'])
    parser.add_argument('--inv_gamma_beta', type=float, default=defaults_dict['inv_gamma_beta'])
    parser.add_argument('--gate_offset', type=float, default=defaults_dict['gate_offset'])
    parser.add_argument('--resid_init_std', type=float, default=defaults_dict['resid_init_std'])
    parser.add_argument('--use_wandb', type=bool, default=defaults_dict['use_wandb'])
    parser.add_argument('--infer_inv_gamma', type=bool, default=defaults_dict['infer_inv_gamma'])
    parser.add_argument('--use_mvg', type=bool, default=defaults_dict['use_mvg'])
    parser.add_argument('--uniform_prop', type=float, default=defaults_dict['uniform_prop'])
    parser.add_argument('--use_swa', type=bool, default=defaults_dict['use_swa'])

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
        misalign_prob=0.0,
        use_population_prior=False,
        use_wandb=True,
        inv_gamma_alpha=0.0,
        inv_gamma_beta=0.0,
        gate_offset=0.0,
        resid_init_std=1e-1,
        channelwise_gating=True,
        infer_inv_gamma=False,
        use_mvg=False,
        uniform_prop=0.1,
        use_swa=True,
    )
    return defaults


def train_model(config_dict):
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']

    config_dict.no_intermediate_layers = max(1, config_dict.no_intermediate_layers)
    config_dict.no_units = max(1, config_dict.no_units)

    optimiser = tf.keras.optimizers.Adam(learning_rate=config_dict.pt_lr)
    if config_dict.use_swa:
        optimiser = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=config_dict.pt_lr)
        optimiser = tfa.optimizers.SWA(optimiser, start_averaging=100, average_period=10)
    #optimiser = tfa.optimizers.MovingAverage(optimiser)

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
                             channelwise_gating=config_dict.channelwise_gating,
                             infer_inv_gamma=config_dict.infer_inv_gamma,
                             use_population_prior=config_dict.use_population_prior,
                             use_mvg=config_dict.use_mvg
                             )

    model = trainer.create_encoder(gate_offset=config_dict.gate_offset, resid_init_std=config_dict.resid_init_std)

    if True:#not config_dict.use_population_prior:
        def synth_loss(x, y):
            return trainer.synthetic_data_loss(x, y, config_dict.use_r2p_loss, config_dict.inv_gamma_alpha,
                                               config_dict.inv_gamma_beta)

        def oef_metric(x, y):
            return trainer.oef_metric(x, y)

        def dbv_metric(x, y):
            return trainer.dbv_metric(x, y)

        def r2p_metric(x, y):
            return trainer.r2p_metric(x, y)

        def oef_alpha_metric(x, y):
            return y[0, 0, 0, 0, 4]

        def oef_beta_metric(x, y):
            return y[0, 0, 0, 0, 5]

        def dbv_alpha_metric(x, y):
            return y[0, 0, 0, 0, 6]

        def dbv_beta_metric(x, y):
            return y[0, 0, 0, 0, 7]

        metrics = [oef_metric, dbv_metric, r2p_metric]
        if config_dict.infer_inv_gamma:
            metrics.extend([oef_alpha_metric, oef_beta_metric, dbv_beta_metric, dbv_alpha_metric])
        model.compile(optimiser, loss=[synth_loss, None, None],
                      metrics=[metrics, None, None])

        # x, y = load_synthetic_dataset(args.f)
        x, y = create_synthetic_dataset(params, config_dict.full_model, config_dict.use_blood,
                                        config_dict.misalign_prob, uniform_prop=config_dict.uniform_prop)
        synthetic_dataset, synthetic_validation = prepare_synthetic_dataset(x, y)
        model.fit(synthetic_dataset, epochs=config_dict.no_pt_epochs, validation_data=synthetic_validation,
                  callbacks=[tf.keras.callbacks.TerminateOnNaN()])

        del synthetic_dataset
        del synthetic_validation

    if not os.path.exists(config_dict.d):
        raise Exception('Real data directory not found')

    # Load real data for fine-tuning, using the model trained on synthetic data for priors
    ase_data = np.load(f'{config_dict.d}/ASE_scan.npy')
    ase_inf_data = np.load(f'{config_dict.d}/ASE_INF.npy')
    ase_sup_data = np.load(f'{config_dict.d}/ASE_SUP.npy')

    train_data = np.concatenate([ase_data, ase_inf_data, ase_sup_data], axis=0)
    train_dataset = prepare_dataset(train_data, model, config_dict.crop_size)

    hyperv_data = np.load(f'{config_dict.d}/hyperv_ase.npy')
    # Split into data with just a GM mask (for validation loss calculation) and a brain mask (for image generation)
    hyperv_with_brain_mask = np.concatenate([hyperv_data[:, :, :, :, :-2], hyperv_data[:, :, :, :, -1:]], -1)
    hyperv_data = hyperv_data[:, :, :, :, :-1]
    baseline_data = np.load(f'{config_dict.d}/baseline_ase.npy')
    baseline_with_brain_mask = np.concatenate([baseline_data[:, :, :, :, :-2], baseline_data[:, :, :, :, -1:]], -1)
    baseline_data = baseline_data[:, :, :, :, :-1]

    transform_dir_baseline = config_dict.d + '/transforms_baseline/'
    transform_dir_hyperv = config_dict.d + '/transforms_hyperv/'

    study_data = np.concatenate([hyperv_data, baseline_data], axis=0)
    baseline_priors, _, _ = model.predict(baseline_with_brain_mask[:, :, :, :, :-1] * baseline_with_brain_mask[:, :, :, :, -1:])

    hyperv_priors, _, _ = model.predict(hyperv_with_brain_mask[:, :, :, :, :-1] * hyperv_with_brain_mask[:, :, :, :, -1:])

    if trainer._use_mvg:
        baseline_priors = baseline_priors[:, :, :, :, 0:5]
        hyperv_priors = hyperv_priors[:, :, :, :, 0:5]
    else:
        baseline_priors = baseline_priors[:, :, :, :, 0:4]
        hyperv_priors = hyperv_priors[:, :, :, :, 0:4]

    study_dataset = prepare_dataset(study_data, model, 76, training=False)

    # If we're not using the population prior we may want to save predictions from our initial model
    if True:
        trainer.estimate_population_param_distribution(model, baseline_data)

        if config_dict.save_directory is not None:
            if not os.path.exists(config_dict.save_directory):
                os.makedirs(config_dict.save_directory)
            model.save_weights(config_dict.save_directory + '/pt_model.h5')
            trainer.save_predictions(model, baseline_with_brain_mask, config_dict.save_directory + '/pt_baseline',
                                     transform_directory=transform_dir_baseline)
            trainer.save_predictions(model, hyperv_with_brain_mask, config_dict.save_directory + '/pt_hyperv',
                                     transform_directory=transform_dir_hyperv)



    class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate):
            self.initial_learning_rate = initial_learning_rate
            self.steps_per_epoch = 100

        def __call__(self, step):
            const_until = 20 * self.steps_per_epoch
            x_recomp = tf.cast(tf.convert_to_tensor(step), tf.float32)
            c = tf.cast(const_until, x_recomp.dtype.base_dtype)

            op = tf.cast(self.initial_learning_rate, tf.float32) * \
                 tf.pow(tf.cast(0.9, tf.float32), tf.cast((1.0 + (x_recomp - c) / self.steps_per_epoch), tf.float32))
            value = tf.case([(x_recomp > c, lambda: op)], default=lambda: self.initial_learning_rate)
            return value

    if config_dict.use_swa:
        full_optimiser = tfa.optimizers.AdamW(weight_decay=LRSchedule(1e-4),
                                              learning_rate=LRSchedule(config_dict.ft_lr))
        full_optimiser = tfa.optimizers.SWA(full_optimiser)
    else:
        full_optimiser = tf.keras.optimizers.Adam(learning_rate=LRSchedule(config_dict.ft_lr))

    input_3d = keras.layers.Input((None, None, 8, 11))
    input_mask = keras.layers.Input((None, None, 8, 1))
    params['simulate_noise'] = 'False'
    sig_gen_layer = SignalGenerationLayer(params, config_dict.full_model, config_dict.use_blood)
    full_model = trainer.build_fine_tuner(model, sig_gen_layer, input_3d, input_mask)

    kl_var = tf.Variable(1.0, trainable=False)
    def fine_tune_loss(x, y):
        return trainer.fine_tune_loss_fn(x, y)

    def predictions_loss(t, p):
        return trainer.kl_loss(t, p) * kl_var + \
               trainer.smoothness_loss(t, p) * config_dict.smoothness_weight

    def sigma_metric(t, p):
        return tf.reduce_mean(p[:, :, :, :, -1:])

    class ELBOCallback(tf.keras.callbacks.Callback):
        def __init__(self, dataset):
            self._iter = iter(dataset)

        def on_epoch_end(self, epoch, logs=None):
            nll_total = 0.0
            kl_total = 0.0
            smoothness_total = 0.0
            no_batches = 4
            for i in range(no_batches):
                data, y = next(self._iter)
                nll = 0.0
                for i in range(10):
                    predictions = self.model.predict(data)
                    nll += fine_tune_loss(y['predicted_images'], predictions['predicted_images'])
                nll = nll / 10.0
                nll_total = nll + nll_total
                kl_total = kl_total +  trainer.kl_loss(y['predictions'], predictions['predictions'])
                smoothness_total = smoothness_total + trainer.smoothness_loss(y['predictions'], predictions['predictions'])

            nll = nll_total / no_batches
            kl = kl_total / no_batches
            smoothness = smoothness_total / no_batches

            metrics = {'val_nll': nll,
                       'val_elbo': nll + kl,
                       'val_elbo_smooth': nll + kl * kl_var + smoothness * config_dict.smoothness_weight,
                       'val_smoothness': smoothness,
                       'val_smoothness_scaled': smoothness * config_dict.smoothness_weight,
                       'val_kl': kl}

            wandb.log(metrics)

    elbo_callback = ELBOCallback(study_dataset)

    def smoothness_metric(x, y):
        return trainer.smoothness_loss(x, y)

    def kl_metric(x, y):
        return trainer.kl_loss(x, y)

    full_model.compile(full_optimiser,
                       loss={'predicted_images': fine_tune_loss,
                             'predictions': predictions_loss},
                       metrics={'predictions': [smoothness_metric, kl_metric],
                                'predicted_images': sigma_metric})

    callbacks = [WandbCallback(), elbo_callback, tf.keras.callbacks.TerminateOnNaN()]

    full_model.fit(train_dataset, steps_per_epoch=100, epochs=config_dict.no_ft_epochs, callbacks=callbacks)
    trainer.estimate_population_param_distribution(model, baseline_data)

    if config_dict.save_directory is not None:
        if not os.path.exists(config_dict.save_directory):
            os.makedirs(config_dict.save_directory)
        model.save_weights(config_dict.save_directory + '/final_model.h5')
        trainer.save_predictions(model, baseline_with_brain_mask, config_dict.save_directory + '/baseline',
                                 transform_directory=transform_dir_baseline, use_first_op=False, fine_tuner_model=full_model,
                                 priors=baseline_priors)

        trainer.save_predictions(model, hyperv_with_brain_mask, config_dict.save_directory + '/hyperv',
                                 transform_directory=transform_dir_hyperv, use_first_op=False, fine_tuner_model=full_model,
                                 priors=hyperv_priors)


if __name__ == '__main__':
    import sys
    import yaml

    tf.random.set_seed(1)
    np.random.seed(1)

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
