import argparse
import pandas as pd
from dataloader import IHDP_loader
from bartpy.sklearnmodel import SklearnModel

parser = argparse.ArgumentParser(description="BART")
# experiment settings
parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--repeat_num', type=int, default=1000, help='repetition of experiments for estimating confidence intervals')
parser.add_argument('--iter_num', type=int, default=2000, help='maximum gradient steps for each training')
parser.add_argument('--early_freq', type=int, default=200, help='frequencyt of checking early stopping criteria')
parser.add_argument('--data_path', type=str, default='../../data/IHDP/ihdp_npci_1-1000')

# optimizer settings
parser.add_argument('--batch_size', type=int, default=200, help='mini-batch size for each gradient step')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for Adam optimizer')
# parser.add_argument('--betas', type=float, nargs='+', default=(0.9, 0.999), help='beta values for Adam optimizer')
#
# # network architecture settings
# parser.add_argument('--input_norm', type=bool, default=True, help='input normalization')
# parser.add_argument('--output_norm', type=bool, default=True, help='output normalization by modifying final layer bias')
# parser.add_argument('--input_dim', type=int, default=25, help='input dimension size for encoder')
# parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension size for each network')
# parser.add_argument('--enc_layer_num', type=int, default=2, help='number of hidden layers in encoder')
# parser.add_argument('--dec_layer_num', type=int, default=1, help='number of hidden layers in decoder (classifier, regreesor)')
# parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate while training')
#
# # loss settings
# parser.add_argument('--ib_coef', type=float, default=10., help='regularization coef for Information Bottleneck (IB)')
# parser.add_argument('--cpvr_coef', type=float, default=0.1, help='regularization coef for Counterfactual Predictive Variance Regularization (CPVR)')
# parser.add_argument('--particle_num', type=int, default=12, help='Particle numbers of Information bottelenck')
#
# # uncertainty calibration
# parser.add_argument('--drop_type', type=str, default='kl', help='type of drop the samples with high uncertainty (kl, random)')
# parser.add_argument('--drop_ratio', type=float, default=0.0, help='ratio of drop the samples with high uncertainty ')
args = parser.parse_args()

# make dataloaders
train_loader = IHDP_loader(file_name=args.data_path,
                            split_type='train',
                            batch_size=args.batch_size)
valid_loader = IHDP_loader(file_name=args.data_path,
                            split_type='valid',
                            batch_size=args.batch_size)
in_test_loader = IHDP_loader(file_name=args.data_path,
                            split_type='test',
                            in_sample=True)
out_test_loader = IHDP_loader(file_name=args.data_path,
                            split_type='test',
                            in_sample=False)

dataloaders = (train_loader, valid_loader, in_test_loader, out_test_loader)


model = SklearnModel()
"""
The main access point to building BART models in BartPy
Parameters
----------
n_trees: int
    the number of trees to use, more trees will make a smoother fit, but slow training and fitting
n_chains: int
    the number of independent chains to run
    more chains will improve the quality of the samples, but will require more computation
sigma_a: float
    shape parameter of the prior on sigma
sigma_b: float
    scale parameter of the prior on sigma
n_samples: int
    how many recorded samples to take
n_burn: int
    how many samples to run without recording to reach convergence
thin: float
    percentage of samples to store.
    use this to save memory when running large models
p_grow: float
    probability of choosing a grow mutation in tree mutation sampling
p_prune: float
    probability of choosing a prune mutation in tree mutation sampling
alpha: float
    prior parameter on tree structure
beta: float
    prior parameter on tree structure
store_in_sample_predictions: bool
    whether to store full prediction samples
    set to False if you don't need in sample results - saves a lot of memory
store_acceptance_trace: bool
    whether to store acceptance rates of the gibbs samples
    unless you're very memory constrained, you wouldn't want to set this to false
    useful for diagnostics
n_jobs: int
    how many cores to use when computing MCMC samples
    set to `-1` to use all cores
"""

for i in range(args.repeat_num):
    # set data loaders id for repetition
    for loader in dataloaders:
        loader.set_id(i)
    # loader return (x, t, yf, ycf, mu0, mu1)

    # train data load
    train_x, train_t, train_yf, _, _, _ = next(train_loader)

    # validation data load
    valid_x, valid_t, valid_yf, _, _, _ = next(valid_loader)

    # make in-test data for each repetition
    in_test_x, in_test_t, _, _, in_test_mu_0, in_test_mu_1 = next(in_test_loader)

    # make out-test data for each repetition
    out_test_x, out_test_t, _, _, out_test_mu_0, out_test_mu_1 = next(out_test_loader)


    model.fit(train_x, train_yf) # Fit the model
    predictions = model.predict() # Make predictions on the train set
    out_of_sample_ypredictions = model.predict(in_test_x) # Make predictions on new data
