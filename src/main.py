#%%
import numpy as np
import torch
import data_prep as dp
from world import config
from utils import train_model, print_metrics

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

SEED = config['seed']
# Set a random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set global variables
R_BETA = config['r_beta']
R_METHOD = config['r_method']
A_BETA = config['a_beta']
A_METHOD = config['a_method']
DATASET = config['dataset']
VERBOSE = config['verbose']
MODEL = config['model']
NUM_EXP = config['num_exp']
MIN_USER_RATINGS = config['min_u_ratings']
MODEL_OPTION = config['option']
NUM_LAYERS = config['num_layers']
DECAY = config['decay']

# STEP 1: loading dataset
if VERBOSE:
    print(f'loading {DATASET} ...')

df, _, _, stats = dp.load_data(dataset=DATASET, min_interaction_threshold=MIN_USER_RATINGS, verbose=VERBOSE)

NUM_USERS, NUM_ITEMS, MEAN_RATING, NUM_RATINGS, TIME_DISTANCE = stats['num_users'], stats['num_items'], stats['mean_rating'], stats['num_ratings'], stats['time_distance']

# STEP 2: adding absolute and relative decays for users and items
df = dp.add_abs_decay(df, method=A_METHOD, beta=A_BETA, verbose=VERBOSE)
df = dp.add_u_rel_decay(df, method=R_METHOD, beta=R_BETA, verbose=VERBOSE)
#df = dp.add_i_rel_decay(df, method=R_METHOD, beta=R_BETA, verbose=VERBOSE)

stats2 = {'num_users': NUM_USERS, 'num_items': NUM_ITEMS,  'num_interactions': NUM_RATINGS}

# STEP 3: getting the interaction matrix values
rmat_data = dp.get_rmat_values(df, verbose=VERBOSE)
 
seeds = [7, 12, 89, 91, 41]
#seeds = [7]

rmses, recalls, precs, ncdgs_5, ncdgs_10, ncdgs_15, ncdgs_20 = [], [], [], [], [], [], []

exp_n = 1

for seed in seeds:
    print(f'Experiment ({exp_n}) starts with seed:{seed}')
    rmse, recall, prec, ncdg = train_model(model=MODEL, data=rmat_data, data_stats=stats, config=config, t_seed=seed)

    rmses.append(float(rmse))
    recalls.append(float(recall))
    precs.append(float(prec))
    ncdgs_5.append(float(ncdg["@5"]))
    ncdgs_10.append(float(ncdg["@10"]))
    ncdgs_15.append(float(ncdg["@15"]))
    ncdgs_20.append(float(ncdg["@20"]))

    exp_n += 1

print_metrics(rmses, recalls, precs, ncdg, stats=stats2)
