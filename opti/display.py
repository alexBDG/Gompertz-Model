import os
import random
import numpy as np

import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt

from hyperopt import space_eval
from hyperopt.plotting import default_status_colors
from hyperopt.plotting import miscs_to_idxs_vals


# Set seed
random.seed(0)
np.random.seed(0)



def main_plot_history(trials, title="Loss History", directory=None,
                      save_name='hp_history'):
    """Plot a hystory report of the given trial.

    Parameters
    ----------
    trials : Hyperopt object
        Trial object define before calling the `fmin` function.
    title : str, default='Loss History'
        Title of the figure.
    directory : str or None, default=None
        Path where to save the figure.
    save_name : str, default="hp_history"
        File name.
    """

    # Show the un-finished or error trials
    status_colors = default_status_colors
    Ys, colors = zip(
        *[
            (y, status_colors[s])
            for y, s in zip(trials.losses(),
                            trials.statuses())
            if y is not None
        ]
    )

    # Make the plot
    fig = plt.figure()

    # Plot the trials
    plt.scatter(range(len(Ys)), Ys, c=colors)
    plt.xlabel("trials")
    plt.ylabel("loss")
    plt.yscale('log')

    # Plot averaged best error
    best_err = trials.average_best_error()
    plt.axhline(best_err, c="g")

    # Add title
    plt.title(title)

    if directory is not None:
        plt.savefig(os.path.join(directory, save_name+".png"))
        plt.close(fig)
    else:
        plt.show()


def main_plot_histogram(trials, title="Loss Histogram", directory=None,
                        save_name='hp_histogram'):
    """Plot a histogram report of the given trial.

    Parameters
    ----------
    trials : Hyperopt object
        Trial object define before calling the `fmin` function.
    title : str, default='Loss Histogram'
        Title of the figure.
    directory : str or None, default=None
        Path where to save the figure.
    save_name : str, default="hp_histogram"
        File name.
    """

    # Deal with ok vs. un-finished vs. error trials
    status_colors = default_status_colors
    Xs, Ys, Ss, Cs = zip(
        *[
            (x, y, s, status_colors[s])
            for (x, y, s) in zip(trials.specs,
                                 trials.losses(),
                                 trials.statuses())
            if y is not None
        ]
    )

    # Make the plot
    fig = plt.figure()

    # Plot the trials
    logbins = np.logspace(np.log10(np.min(Ys)), np.log10(np.max(Ys)), 20)
    plt.hist(Ys, bins=logbins)
    plt.xlabel("loss")
    plt.ylabel("frequency")
    plt.xscale('log')

    plt.title(title)

    if directory is not None:
        plt.savefig(os.path.join(directory, save_name+".png"))
        plt.close(fig)
    else:
        plt.show()


def f_wrap_space_eval(hp_space, trial):
    """Utility function for more consise optimization history extraction

    Parameters
    ----------
    hp_space : dict
        hyperspace from which points are sampled
    trial : object
        hyperopt.Trials object

    Returns
    -------
    new_space : dict
        Dictonnary of label and values of hyperparameter in trial
    """
    new_space = space_eval(
        hp_space, {
            k: v[0] for (k, v) in trial['misc']['vals'].items() if len(v) > 0
        }
    )

    return new_space


def f_unpack_dict(dct):
    """Unpacks all sub-dictionaries in given dictionary recursively. There
    should be no duplicated keys across all nested subdictionaries, or some
    instances will be lost without warning.

    Parameters
    ----------
    dct : dict
        Dictionary to unpack.

    Returns
    -------
    res : dict
        Unpacked dictionary
    """

    res = {}
    for (k, v) in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v

    return res


def main_plot_vars(trials, fontsize=10, columns=5, arrange_by_loss=False,
                   space=None, directory=None, save_name='hp_vars'):
    """Plot a full variables report of the given trial.

    Parameters
    ----------
    trials : Hyperopt object
        Trial object define before calling the `fmin` function.
    fontsize : int, default=10
        Fontsize of the subgraphics.
    columns : int, default=5
        Number of columns to plot.
    arrange_by_loss : bool, default=False
        Tick labels option.
    space : hyperopt.pyll.Apply node or None, default=None
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp_<xxx> nodes
        (see `hyperopt.hp` and `hyperopt.pyll_utils`).
    directory : str or None, default=None
        Path where to save the figure.
    save_name : str, default="hp_histogram"
        File name.
    """

    # Get results
    idxs, vals = miscs_to_idxs_vals(trials.miscs)

    if space is not None:
        samples = [
            f_unpack_dict(f_wrap_space_eval(space, x))
            for x in trials.trials
        ]
        samples = [
            {key: [sample[key] for sample in samples]}
            for key in samples[0].keys()
        ]
        dict_samples = {}
        for sub_dict in samples:
            dict_samples = {**dict_samples, **sub_dict}

        # Update results
        vals = dict_samples


    # Get losses list
    losses = trials.losses()

    # Find Min/Max
    finite_losses = [y for y in losses if y not in (None, float("inf"))]
    min_loss = np.min(finite_losses)
    max_loss = np.max(finite_losses)

    argmin_loss = np.argmin([
        y if y not in (None, float("inf")) else 0.0 for y in losses
    ])
    # Define a different marker for best loss
    m = np.full(shape=(len(losses)), fill_value='o')
    m[argmin_loss] = '*'

    # Link every loss to its trial iteration
    loss_by_tid = dict(zip(trials.tids, losses))


    all_labels = list(idxs.keys())
    titles = all_labels
    order = np.argsort(titles)

    n_col = min(columns, len(all_labels)+1) # +1 for legend
    n_row = int(np.ceil((len(all_labels)+1) / float(n_col))) # +1 for legend

    # Make the plot
    fig = plt.figure(figsize=(1+n_col*3, 1+n_row*3))

    for plotnum, varnum in enumerate(order):
        label = all_labels[varnum]
        plt.subplot(n_row, n_col, plotnum + 1)

        # hide x ticks
        plt.xticks([], [])

        dist_name = label

        # Define x value
        if arrange_by_loss:
            x = [loss_by_tid[ii] for ii in idxs[label]]
        else:
            x = idxs[label]

        # Define y value
        sign = 1 if np.max(vals[label])>0 else -1
        y = np.log(np.array(vals[label]) * sign) * sign

        # Name of the current variable
        plt.title(titles[varnum], fontsize=fontsize)
        c = list([loss_by_tid[ii] for ii in idxs[label]])
        if len(y):
            sct = plt.scatter(
                x, y, c=c, cmap="coolwarm", norm=matplotlib.colors.LogNorm()
            )
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                    marker_obj.get_transform()
                )
                paths.append(path)
            sct.set_paths(paths)

        nums, texts = plt.yticks()
        plt.yticks(nums, ["{:.1e}".format(sign*np.exp(t)) for t in nums])

    # Plot legend
    ax = plt.subplot(n_row, n_col, plotnum + 2, frameon=False)
    ax.tick_params(
        labelbottom=False, labeltop=False, labelleft=False,
        labelright=False, labelcolor='none', top=False,
        bottom=False, left=False, right=False
    )
    # ax.set_xscale('log')
    cbar = plt.colorbar(mappable=sct, ax=ax, orientation='horizontal', aspect=5)
    cbar.set_label('Loss')

    plt.scatter(
        [0],[0.5], c=np.mean(finite_losses), marker='o',
        cmap="coolwarm", label='Other loss'
    )#.set_visible(False)
    plt.scatter(
        [0],[0], c=min_loss, marker='*',
        cmap="coolwarm", label='Best loss'
    )#.set_visible(False)
    plt.legend(title='Legend')

    fig.tight_layout()

    if directory is not None:
        plt.savefig(os.path.join(directory, f"{save_name}.png"))
        plt.close(fig)
    else:
        plt.show()
