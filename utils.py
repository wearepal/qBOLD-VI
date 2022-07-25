import argparse


def setup_argparser(defaults_dict):
    parser = argparse.ArgumentParser(description='Train neural network for parameter estimation')
    parser.add_argument('-d', default='/home/data/qbold/', help='path to the real data directory')
    parser.add_argument('-f', default='synthetic_data.npz', help='path to synthetic data file')
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
    parser.add_argument('--infer_inv_gamma', type=bool, default=defaults_dict['infer_inv_gamma'])
    parser.add_argument('--use_mvg', type=bool, default=defaults_dict['use_mvg'])
    parser.add_argument('--uniform_prop', type=float, default=defaults_dict['uniform_prop'])
    parser.add_argument('--use_swa', type=bool, default=defaults_dict['use_swa'])
    parser.add_argument('--adamw_decay', type=float, default=defaults_dict['adamw_decay'])
    parser.add_argument('--pt_adamw_decay', type=float, default=defaults_dict['pt_adamw_decay'])
    parser.add_argument('--predict_log_data', type=bool, default=defaults_dict['predict_log_data'])
    parser.add_argument('--wandb_project', default=defaults_dict['wandb_project'])

    return parser


# These defaults would be used as a basis, and values here will be replaced if optimal.yaml is used
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
        use_population_prior=True,
        wandb_project='',
        inv_gamma_alpha=0.0,
        inv_gamma_beta=0.0,
        gate_offset=0.0,
        resid_init_std=1e-1,
        channelwise_gating=True,
        infer_inv_gamma=False,
        use_mvg=False,
        uniform_prop=0.1,
        use_swa=True,
        adamw_decay=2e-4,
        pt_adamw_decay=2e-4,
        predict_log_data=True
    )
    return defaults


def load_arguments():
    import yaml
    import sys

    if (len(sys.argv) >= 2) and (".yaml" in sys.argv[1]):
        # Read the yaml filename
        yaml_file = sys.argv[1]
        """if (len(sys.argv) >= 3) and isinstance(sys.argv[2], str):
            d = sys.argv[2]
        if (len(sys.argv) == 6):
            try:
                tau_start = float(sys.argv[3])
                tau_end = float(sys.argv[4])
                tau_step = float(sys.argv[5])
            except ValueError:
                print("Incorrect values provided for tau start end and step")"""
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

    """if tau_start is not None and tau_step is not None and tau_end is not None:
        args['tau_start'] = tau_start
        args['tau_end'] = tau_end
        args['tau_step'] = tau_step"""

    return args
