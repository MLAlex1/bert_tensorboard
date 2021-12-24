from torch.utils.data import Dataset
import torch

import copy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_grad_flow(named_parameters : iter, skip_prob : float = 0.5, verbose : bool = False, seed : int = 0) -> plt.figure:
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    skip_prob : skip some random layers for better visualization if there are a lot of layers
    seed : random seed to skip layers
    '''

    np.random.seed(seed)
    name_replace = {"encoder" : "enc", "layer" : "l"}
    plt.rcParams["figure.figsize"] = (20, 12)
    # plt.rcParams['figure.dpi']= 150

    mean_grads, max_grads, zero_grads, stds = [], [], [], []
    name_layers = []
    for (n, p) in (named_parameters):
        if (p.requires_grad) and ("bias" not in n):
            if np.random.rand() < skip_prob:
                if verbose:
                    print(f"skipped {n}")
                continue
            name_layers.append('.'.join([name_replace.get(el, el) for el in n.split(".") if el not in ['weight', 'bert', 'self']]))
            mean_grads.append(p.grad.abs().mean().detach().cpu().item())
            max_grads.append(p.grad.abs().max().detach().cpu().item())
            stds.append(p.grad.abs().std().detach().cpu().item())
            zero_grads.append(torch.sum(p.grad.abs() == 0.).detach().cpu().item() / p.grad.nelement())
        elif not (p.requires_grad):
            print(f"{n} does not require grad")

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set(font_scale = 1.2)
    fig, ax = plt.subplots(3)
    sns.barplot(x=name_layers, y=max_grads, palette=['b']*len(name_layers), alpha=0.9, ax=ax[2], label="max grads",)
    sns.barplot(x=name_layers, y=mean_grads, palette=['r']*len(name_layers), alpha=0.9, ax=ax[2], label="mean grads",)

    sns.barplot(x=name_layers, y=zero_grads, palette=['k']*len(name_layers), alpha=0.9, ax=ax[0], label="percentage zero grads")

    sns.barplot(x=name_layers, y=stds, palette=['c']*len(name_layers), alpha=0.9, ax=ax[1], label="standard dev grads")

    ax[2].set_ylim([-0.005, 0.05])
    ax[1].set_ylim([-0.005, 0.1])
    ax[0].set_ylim([-0.005, 1.05])
    ax[2] = ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation = 90)
    ax[0].set(xticklabels=[])
    ax[1].set(xticklabels=[])


    ax[0].legend()
    ax[1].legend()
    plt.legend()
    plt.close()
    return fig

def plot_ratios(ratios : dict,
                skip_prob : float = 0.5,
                verbose : bool = False,
                seed : int = 0) -> plt.figure:
    '''Plots the update/param ratio.
    Can be used for checking for possible gradient vanishing / exploding problems.

    This ratio should be around 1e-3.
    If it is lower than this then the learning rate might be too low.
    If it is higher then the learning rate is likely too high

    skip_prob : skip some random layers for better visualization if there are a lot of layers
    seed : random seed to skip layers
    '''
    np.random.seed(seed)
    name_replace = {"encoder" : "enc", "layer" : "l"}
    plt.rcParams["figure.figsize"] = (20, 12)
    # plt.rcParams['figure.dpi']= 150

    ratios_list = []
    name_layers = []
    for (n, r) in (ratios.items()):
        if np.random.rand() < skip_prob:
            if verbose:
                print(f"skipped {n}")
            continue
        name_layers.append('.'.join([name_replace.get(el, el) for el in n.split(".") if el not in ['weight', 'bert', 'self']]))
        ratios_list.append(r)

    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set(font_scale = 1.2)
    fig, ax = plt.subplots()
    sns.barplot(x=name_layers, y=ratios_list, palette=['b']*len(name_layers), ax=ax, label="ratio update/param",)

    ax.set_ylim([-0.0005, 0.0015])
    ax = ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

    plt.legend()
    plt.close()
    return fig

def compute_ratios(param_prev : dict ,
                   param_next : dict,
                   named_params : dict,
                   ignore_bias : bool = True) -> dict:
    """ Compute update/param ratio
    """
    updates = {}
    param_names = []
    for i, (name, val) in enumerate(named_params.items()):
        param_names.append(name)
        updates[name] = copy.deepcopy((param_next[i] - param_prev[i]).cpu().detach().numpy())

    ratio_updates = {}
    for n in param_names:
        if ignore_bias:
            if 'bias' in n: continue
        k = named_params[n]
        param_scale = np.linalg.norm(k.ravel())
        update = updates[n]
        update_scale = np.linalg.norm(update.ravel())
        ratio_updates[n] = update_scale/param_scale
    return ratio_updates

def copy_model_params(named_params : dict) -> dict:
    """ Copy named parameters of the model
    """
    copy_params = {}
    for (name, val) in (named_params):
        copy_params[name] = copy.deepcopy(val.cpu().detach().numpy())
    return copy_params


class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)
