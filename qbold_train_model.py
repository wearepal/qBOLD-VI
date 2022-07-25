#!/usr/bin/env python3

import os
import numpy as np
from model import EncoderTrainer
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbCallback

from qbold_build_model import ModelBuilder, WeightStatus
from signals import SignalGenerationLayer


class ModelTrainer(ModelBuilder):
    def __init__(self, config_dict, system_params=None):
        super().__init__(config_dict, system_params)
        tf.random.set_seed(1)
        np.random.seed(1)

        if self.config_dict['wandb_project'] != '':
            wandb.init(project=self.config_dict.wandb_project)
            if self.config_dict.get('name') is None:
                wandb.run.name = args['name']

    def train_on_synthetic_data(self):
        from signals import create_synthetic_dataset
        # Check the model is not already fully trained
        assert self.weight_status != WeightStatus.FULL_TRAINED

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config_dict['pt_lr'])
        if self.config_dict['use_swa']:
            optimiser = tfa.optimizers.AdamW(weight_decay=self.config_dict['pt_adamw_decay'],
                                             learning_rate=self.config_dict['pt_lr'])
            optimiser = tfa.optimizers.SWA(optimiser, start_averaging=22 * 40, average_period=22)

        def synth_loss(_x, _y):
            return self.trainer.synthetic_data_loss(_x, _y, self.config_dict['use_r2p_loss'],
                                                    self.config_dict['inv_gamma_alpha'],
                                                    self.config_dict['inv_gamma_beta'])

        def oef_metric(_x, _y):
            return self.trainer.oef_metric(_x, _y)

        def dbv_metric(_x, _y):
            return self.trainer.dbv_metric(_x, _y)

        def r2p_metric(_x, _y):
            return self.trainer.r2p_metric(_x, _y)

        def oef_alpha_metric(_x, _y):
            return _y[0, 0, 0, 0, 4]

        def oef_beta_metric(_x, _y):
            return _y[0, 0, 0, 0, 5]

        def dbv_alpha_metric(_x, _y):
            return _y[0, 0, 0, 0, 6]

        def dbv_beta_metric(_x, _y):
            return _y[0, 0, 0, 0, 7]

        metrics = [oef_metric, dbv_metric, r2p_metric]
        if self.config_dict['infer_inv_gamma']:
            metrics.extend([oef_alpha_metric, oef_beta_metric, dbv_beta_metric, dbv_alpha_metric])
        self.model.compile(optimiser, loss=[synth_loss, None, None],
                           metrics=[metrics, None, None])

        x, y = create_synthetic_dataset(self.system_params, self.config_dict['full_model'],
                                        self.config_dict['use_blood'],
                                        self.config_dict['misalign_prob'],
                                        uniform_prop=self.config_dict['uniform_prop'])
        synthetic_dataset, synthetic_validation = ModelTrainer.prepare_synthetic_dataset(x, y)
        self.model.fit(synthetic_dataset, epochs=self.config_dict['no_pt_epochs'], validation_data=synthetic_validation,
                       callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        self.model.save_weights(self.pt_model_weights)
        # Delete these dataset objects to save space
        del synthetic_dataset
        del synthetic_validation

        self.weight_status = WeightStatus.PRE_TRAINED

    @staticmethod
    def prepare_synthetic_dataset(x, y):
        train_conv = True
        # If we're building a convolutional model, reshape the synthetic data to look like images, note we only do
        # 1x1x1 convs for pre-training
        if train_conv:
            # Reshape to being more image like for layer normalisation (if we use this)
            x = np.reshape(x, (-1, 10, 10, 5, x.shape[-1]))
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
        synthetic_dataset = synthetic_dataset.batch(512)
        return synthetic_dataset, (valid_x, valid_y)

    def train_model(self):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        if self.weight_status == WeightStatus.NOT_TRAINED:
            self.train_on_synthetic_data()
            print('saved', self.pt_model_weights)

        if self.weight_status == WeightStatus.PRE_TRAINED:
            no_taus = len(np.arange(float(self.system_params['tau_start']), float(self.system_params['tau_end']),
                                    float(self.system_params['tau_step'])))
            input_3d = keras.layers.Input((None, None, 8, no_taus))
            input_mask = keras.layers.Input((None, None, 8, 1))
            self.system_params['simulate_noise'] = 'False'
            # Generate
            sig_gen_layer = SignalGenerationLayer(self.system_params, self.config_dict['full_model'],
                                                  self.config_dict['use_blood'])

            full_model = self.trainer.build_fine_tuner(self.model, sig_gen_layer, input_3d, input_mask)

            data_directory = self.config_dict['d']
            hyperv_dir = f'{data_directory}/hyperv_ase.npy'
            hyperv_data = self.load_condition_data(hyperv_dir, False)
            baseline_dir = f'{data_directory}/baseline_ase.npy'
            baseline_data = self.load_condition_data(baseline_dir, False)
            study_data = np.concatenate([hyperv_data, baseline_data], axis=0)
            study_dataset = self.prepare_dataset(study_data, self.model, 76, training=False)

            train_data = self.load_real_data()
            train_dataset = self.prepare_dataset(train_data, self.model, self.config_dict['crop_size'])

            self.train_full_model(self.trainer, full_model, study_dataset, train_dataset)

            self.trainer.estimate_population_param_distribution(self.model, baseline_data)

            self.model.save_weights(self.final_model_weights)

            del train_dataset
            del study_dataset

    def load_real_data(self):
        if not os.path.exists(self.config_dict['d']):
            raise Exception('Real data directory not found')

        data_directory = self.config_dict['d']
        # Load real data for fine-tuning, using the model trained on synthetic data for priors
        ase_data = np.load(f'{data_directory}/ASE_scan.npy')
        ase_inf_data = np.load(f'{data_directory}/ASE_INF.npy')
        ase_sup_data = np.load(f'{data_directory}/ASE_SUP.npy')

        return np.concatenate([ase_data, ase_inf_data, ase_sup_data], axis=0)

    def load_condition_data(self, condition_dir, with_brain_mask):
        condition_data = np.load(condition_dir)
        condition_with_brain_mask = np.concatenate(
            [condition_data[:, :, :, :, :-2], condition_data[:, :, :, :, -1:]], -1)
        if with_brain_mask:
            return condition_with_brain_mask[:, :, :, :, :-1] * condition_with_brain_mask[:, :, :, :, -1:]
        return condition_data[:, :, :, :, :-1]

    def prepare_dataset(self, real_data, model, crop_size=20, training=True, blank_crop=True):
        if blank_crop:
            # Prepare the real data, crop out more in the x-dimension to avoid risk of lots of empty voxels
            real_data = np.float32(real_data[:, 17:-17, 10:-10, :, :])
        else:
            real_data = np.float32(real_data)

        _crop_size = [min(crop_size, real_data.shape[1]), min(crop_size, real_data.shape[2])]
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

            # Concatenate to crop
            crop_data = tf.concat([data, predicted_distribution], -1)
            crop_data = tf.image.random_crop(value=crop_data, size=_crop_size + crop_data.shape[-1:])

            # Separate out data and predicted distribution again
            predicted_distribution = crop_data[:, :, -predicted_distribution.shape.as_list()[-1]:]
            predicted_distribution = tf.reshape(predicted_distribution,
                                                _crop_size + predicted_distribution_shape[-2:])

            data = crop_data[:, :, :data.shape[-1]]
            data = tf.reshape(data, _crop_size + data_shape[-2:])
            mask = data[:, :, :, -1:]

            data = data[:, :, :, :-1] * data[:, :, :, -1:]
            # Concatenate the mask unto the data structure
            data = tf.concat([data, mask], -1)

            predicted_distribution = tf.concat([predicted_distribution, mask], -1)

            return (data[:, :, :, :-1], mask), {'predictions': predicted_distribution, 'predicted_images': data}

        real_dataset = real_dataset.map(map_func2)
        real_dataset = real_dataset.repeat(-1)
        if training:
            real_dataset = real_dataset.shuffle(10000)
            real_dataset = real_dataset.batch(38, drop_remainder=True)
        else:
            real_dataset = real_dataset.batch(3, drop_remainder=True)

        return real_dataset

    def train_full_model(self, trainer, full_model, study_dataset, train_dataset):
        assert isinstance(trainer, EncoderTrainer)
        config = self.config_dict

        class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_learning_rate):
                self.initial_learning_rate = initial_learning_rate
                self.steps_per_epoch = 100

            def __call__(self, step):
                const_until = 0.0 * self.steps_per_epoch
                x_recomp = tf.cast(tf.convert_to_tensor(step), tf.float32)
                c = tf.cast(const_until, x_recomp.dtype.base_dtype)

                op = tf.cast(self.initial_learning_rate, tf.float32) * \
                     tf.pow(tf.cast(0.9, tf.float32),
                            tf.cast((1.0 + (x_recomp - c) / self.steps_per_epoch), tf.float32))

                final_lr = self.initial_learning_rate / 1e2
                linear_rate = (final_lr - self.initial_learning_rate) / (40.0 * self.steps_per_epoch - const_until)
                op = self.initial_learning_rate + linear_rate * (x_recomp - c)

                value = tf.case([(x_recomp > c, lambda: op)], default=lambda: self.initial_learning_rate)

                return value

        if self.config_dict['adamw_decay'] > 0.0:
            full_optimiser = tfa.optimizers.AdamW(weight_decay=LRSchedule(self.config_dict['adamw_decay']),
                                                  learning_rate=LRSchedule(self.config_dict['ft_lr']), beta_2=0.9)
        else:
            full_optimiser = tf.keras.optimizers.Adam(learning_rate=LRSchedule(self.config_dict['ft_lr']))
        kl_var = tf.Variable(1.0, trainable=False)

        def fine_tune_loss(x, y):
            return trainer.fine_tune_loss_fn(x, y)

        def predictions_loss(t, p):
            return trainer.kl_loss(t, p) * kl_var + \
                   trainer.smoothness_loss(t, p) * self.config_dict['smoothness_weight']

        def sigma_metric(t, p):
            return tf.reduce_mean(p[:, :, :, :, -1:])

        class ELBOCallback(tf.keras.callbacks.Callback):
            def __init__(self, dataset, config_dict):
                self._iter = iter(dataset)
                self.config_dict = config_dict

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
                    kl_total = kl_total + trainer.kl_loss(y['predictions'], predictions['predictions'])
                    smoothness_total = smoothness_total + trainer.smoothness_loss(y['predictions'],
                                                                                  predictions['predictions'])

                nll = nll_total / no_batches
                kl = kl_total / no_batches
                smoothness = smoothness_total / no_batches

                metrics = {'val_nll': nll,
                           'val_elbo': nll + kl,
                           'val_elbo_smooth': nll + kl * kl_var + smoothness * self.config_dict['smoothness_weight'],
                           'val_smoothness': smoothness,
                           'val_smoothness_scaled': smoothness * self.config_dict['smoothness_weight'],
                           'val_kl': kl}
                if self.config_dict['wandb_project'] != '':
                    wandb.log(metrics)

        elbo_callback = ELBOCallback(study_dataset, self.config_dict)

        def smoothness_metric(x, y):
            return trainer.smoothness_loss(x, y)

        def kl_metric(x, y):
            return trainer.kl_loss(x, y)

        def kl_samples_metric(x, y):
            return trainer.mvg_kl_samples(x, y)

        full_model.compile(full_optimiser,
                           loss={'predicted_images': fine_tune_loss,
                                 'predictions': predictions_loss},
                           metrics={'predictions': [smoothness_metric, kl_metric],
                                    'predicted_images': sigma_metric})
        callbacks = [elbo_callback, tf.keras.callbacks.TerminateOnNaN()]
        if self.config_dict['wandb_project'] != '':
            callbacks.append(WandbCallback())
        full_model.fit(train_dataset, steps_per_epoch=100, epochs=self.config_dict['no_ft_epochs'], callbacks=callbacks)


if __name__ == '__main__':
    from utils import load_arguments

    yaml_file = None
    tau_start = None
    tau_step = None
    tau_end = None
    # If we have a single argument and it's a yaml file, read the config from there
    args = load_arguments()
    print(args)
    model_trainer = ModelTrainer(args)
    model_trainer.train_model()
