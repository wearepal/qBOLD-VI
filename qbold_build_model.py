#!/usr/bin/env python3

import os
import numpy as np
import configparser
from model import EncoderTrainer

from enum import Enum


class WeightStatus(Enum):
    NOT_TRAINED = 0
    PRE_TRAINED = 1
    FULL_TRAINED = 2


class ModelBuilder:
    def __init__(self, config_dict, system_params=None):
        self.config_dict = config_dict
        if system_params:
            self.system_params = system_params
        else:
            self.system_params = ModelBuilder.get_params()

        model, inner_model, trainer = ModelBuilder.create_encoder_model(self.config_dict, self.system_params)
        self.model = model
        self.inner_model = inner_model
        self.trainer = trainer
        self.save_dir = os.path.join(os.getcwd(), self.config_dict['save_directory'])
        # final weights are trained on real data
        self.final_model_weights = os.path.join(self.save_dir, 'final_model.h5')
        # pre-trained weights    are trained on synthetic data
        self.pt_model_weights = os.path.join(self.save_dir, 'pt_model.h5')

        self.weight_status = self.load_model_weights()


    @staticmethod
    def get_params():
        # this method loads the configs expressed in optimal.yaml
        config = configparser.ConfigParser()
        config.read('config')
        return config['DEFAULT']

    def load_model_weights(self):
        # Check if model weights exist in the save directory
        if os.path.isfile(self.final_model_weights):
            self.model.load_weights(self.final_model_weights)
            return WeightStatus.FULL_TRAINED

        elif os.path.isfile(self.pt_model_weights):
            self.model.load_weights(self.pt_model_weights)
            return WeightStatus.PRE_TRAINED
        else:
            print('Model weights do not exit')
            return WeightStatus.NOT_TRAINED

    # This method creates uses EncoderTrainer class defined inside model.py in order to generate a model
    @staticmethod
    def create_encoder_model(config_dict, params):
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
        taus = np.array([0.000, 0.016, 0.020, 0.024, 0.028, 0.032, 0.036, 0.040, 0.044, 0.048, 0.052])#np.arange(float(params['tau_start']), float(params['tau_end']), float(params['tau_step']))
        model, inner_model = trainer.create_encoder(gate_offset=config_dict['gate_offset'],
                                                    resid_init_std=config_dict['resid_init_std'],
                                                    no_ip_images=len(taus))
        return model, inner_model, trainer
