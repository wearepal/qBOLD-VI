# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: #Enter feature name here

import sys
import yaml
import tensorflow as tf
import numpy as np
from train import  setup_argparser, get_defaults, train_model
import wandb

if __name__ == '__main__':
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

    layers_range = np.arange(2, 11, 2)
    units_range = np.arange(2, 11, 2)
    orig_name = args['name']
    for no_layers in layers_range:
        for no_units in units_range:
            for run_idx in range(3):
                args['no_intermediate_layers'] = no_layers
                args['no_units'] = no_units
                args['name'] = orig_name + '-' + str(no_layers) + '-' + str(no_units) + '-' + str(run_idx)
                wandb.init(project='qbold_inference', entity='kasiamoj')
                wandb.run.name = args['name']
                wandb.config.update(args)
                train_model(wandb.config)
                wandb.finish()
