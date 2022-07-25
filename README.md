# qBOLD-VI
This codebase provides a method for learning an amortized variational inference model for inferring the Oxygen Extraction Fraction and Deoxygenated Blood Volume from qBOLD data.

The code is arranged such that:
* signals.py generates the synthetic data, and contains the forward model for the observed qBOLD data.
* model.py: Contains all of the machine learning model specification. 
* train.py: Contains the functionality to train the model, requires a .yaml configuration file as input 
* data_preprocessing.py, pre-processes the qBOLD images into numpy arrays. This will need to be adapted to new datasets.

Please note that I am currently refactoring this code to decouple some elements of the model, particularly the logit-Normal distribution. Please get in touch if you find any issues.

An example Colab notebook that loads a pretrained model is available [here](https://colab.research.google.com/drive/1zJ6yu5-sr-wD4aKWpfFEGlkRGYJDaIuC?usp=sharing), although it should be noted that data is not yet public.

The training code uses [weights and biases (wandb)](wandb.ai) to track experiments and 
perform hyper-parameter optimisation.
The hyper-parameters are specified in YAML files, an example using the optimal configuration is provided in 
configurations/optimal.yaml

The model is described in the preprint "Flexible Amortized Variational Inference in qBOLD MRI" here [https://arxiv.org/abs/2203.05845](https://arxiv.org/abs/2203.05845)
