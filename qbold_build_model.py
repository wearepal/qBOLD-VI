#!/usr/bin/env python3

import os
import numpy as np
import configparser
from model import EncoderTrainer
import tensorflow as tf
import tensorflow_addons as tfa
from signals import create_synthetic_dataset

class ModelBuilder:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def get_params(self):
        # this method loads the configs expressed in optimal.yaml
        config = configparser.ConfigParser()
        config.read('config')
        return config['DEFAULT']

    def build_model(self):
        params = self.get_params()

        if not os.path.isdir(self.config_dict['save_directory']):
            mkdir = 'mkdir optimal'
            os.system(mkdir)

        pt_model_weights = self.config_dict['save_directory'] + '/pt_model.h5'

        model, inner_model, trainer = self.create_encoder_model(self.config_dict, params)

        if not os.path.isfile(pt_model_weights):
            model = self.create_and_train_on_synthetic_data(model, inner_model, trainer, params)
            model.save_weights(pt_model_weights)

    def create_and_train_on_synthetic_data(self, model, inner_model, trainer, params):
        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config_dict['pt_lr'])
        if self.config_dict['use_swa']:
            optimiser = tfa.optimizers.AdamW(weight_decay=self.config_dict['pt_adamw_decay'], learning_rate=self.config_dict['pt_lr'])
            optimiser = tfa.optimizers.SWA(optimiser, start_averaging=22 * 40, average_period=22)
        if self.config_dict['use_population_prior']:
            def synth_loss(x, y):
                return trainer.synthetic_data_loss(x, y, self.config_dict['use_r2p_loss'], self.config_dict['inv_gamma_alpha'],
                                                   self.config_dict['inv_gamma_beta'])

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
            if self.config_dict['infer_inv_gamma']:
                metrics.extend([oef_alpha_metric, oef_beta_metric, dbv_beta_metric, dbv_alpha_metric])
            model.compile(optimiser, loss=[synth_loss, None, None],
                          metrics=[metrics, None, None])

            x, y = create_synthetic_dataset(params, self.config_dict['full_model'], self.config_dict['use_blood'],
                                            self.config_dict['misalign_prob'], uniform_prop=self['config_dict.uniform_prop'])
            synthetic_dataset, synthetic_validation = self.prepare_synthetic_dataset(x, y)
            model.fit(synthetic_dataset, epochs=self.config_dict['no_pt_epochs'], validation_data=synthetic_validation,
                      callbacks=[tf.keras.callbacks.TerminateOnNaN()])

            del synthetic_dataset
            del synthetic_validation
        return model, inner_model, trainer

    def prepare_synthetic_dataset(self, x, y):
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

    # This method creates uses EncoderTrainer class defined inside model.py in order to generate a model
    def create_encoder_model(self, config_dict, params):
        config_dict['no_intermediate_layers'] = max(1, config_dict['no_intermediate_layers'])
        config_dict['no_units'] = max(1, config_dict['no_units'])
        trainer = EncoderTrainer(system_params=params,
                                 no_units=config_dict['no_units'],
                                 use_layer_norm=config_dict['use_layer_norm'],
                                 dropout_rate=config_dict['dropout_rate'],
                                 no_intermediate_layers=config_dict['no_intermediate_layers'],
                                 student_t_df=config_dict['student_t_df'],
                                 initial_im_sigma=config_dict['im_loss_sigma'],
                                 activation_type=config_dict['activation'],
                                 multi_image_normalisation=config_dict['multi_image_normalisation'],
                                 channelwise_gating=config_dict['channelwise_gating'],
                                 infer_inv_gamma=config_dict['infer_inv_gamma'],
                                 use_population_prior=config_dict['use_population_prior'],
                                 use_mvg=config_dict['use_mvg'],
                                 predict_log_data=config_dict['predict_log_data']
                                 )
        taus = np.arange(float(params['tau_start']), float(params['tau_end']), float(params['tau_step']))
        model, inner_model = trainer.create_encoder(gate_offset=config_dict['gate_offset'],
                                                    resid_init_std=config_dict['resid_init_std'], no_ip_images=len(taus))
        return model, inner_model, trainer


