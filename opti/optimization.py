# System modules
import random
import numpy as np

# Optimization module
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import STATUS_OK
from hyperopt import space_eval
from hyperopt import STATUS_FAIL

# Figures modules
from .utils import Gompertz


# Set seed
random.seed(0)
np.random.seed(0)



def get_search_space(k_min=0, k_max=1000, alpha_min=0., alpha_max=1.,
                     shift_min=0., shift_max=200.):
    """Create the bayesian research's space.

    Parameters
    ----------
    k_min : float, default=0
        Lower bound of `K` to use.
    k_max : float, default=1000
        Upper bound of `K` to use.
    alpha_min : float, default=0.
        Lower bound of `alpha` to use.
    alpha_max : float, default=1.
        Upper bound of `alpha` to use.
    shift_min : float, default=0.
        Lower bound of `shift` to use.
    shift_max : float, default=1.
        Upper bound of `shift` to use.

    Returns
    -------
    space : dict
        All arguments with their reseach's spaces.

    Examples
    --------
    >>> # Choose in a finite list
    >>> space = {'arg1': hp.choice('arg1', [1, 2, 3])}
    >>>
    >>> # Choose in uniform integer distribution
    >>>
    >>> # Choose in log normal distribution
    >>> space = {'arg3': hp.lognormal('arg3', 1e-2, 1)}
    >>>
    >>> # Choose in log uniform distribution
    >>> space = {'arg4': hp.loguniform('arg4', np.log(1e-5), np.log(1e-1))}
    """

    space = {
        "k": hp.uniform("k", k_min, k_max),
        "alpha": hp.loguniform("alpha", np.log(alpha_min), np.log(alpha_max)),
        "shift": hp.uniform("shift", shift_min, shift_max),
    }

    return space


def optimize(t_train, f_train, t_phy, space, f_0, history,
             add_physical_loss=False, max_evals=100):
    """Run HyperOpt optimization function.

    Parameters
    ----------
    t_train : ndarray
        Training time data.
    f_train : ndarray
        Training Gompertz data.
    t_phy : ndarray
        Physical time data.
    space : dict
        All arguments with their reseach's spaces.
    f_0 : float
        Initial Gompertz value.
    history : HistoryManager
        Keep trace of all data.
    add_physical_loss : bool, default=False
        If `True`, total loss consider physical one.
    max_evals : int, default=100
        Number maximum of evaluations.

    Returns
    -------
    dict
        Best arguments found by the optimization procedure.
    Trials
        Database interface of the optimization, managed by HyperOpt.
    """

    def objective(args, f_0=f_0):
        k = args.get("k")
        alpha = args.get("alpha")
        shift = args.get("shift")

        f = Gompertz(k=k, alpha=alpha, shift=shift, f_0=f_0)
        f_train_pred = f(t_train)
        f_phy_pred = f(t_phy)
        dfdt_phy_pred = np.gradient(t_phy, t_phy[1]-t_phy[0])

        # Main part - evaluation
        loss_data = np.mean(np.square(
            f_train_pred - f_train
        )) / t_train.shape[0]
        loss_phy = np.mean(np.square(
            dfdt_phy_pred - alpha * f_phy_pred * np.log(k / f_phy_pred)
        )) / t_phy.shape[0]

        loss = loss_data
        if add_physical_loss:
            loss += loss_phy

        history.update(loss, loss_data, loss_phy, k, alpha, shift)

        if np.isnan(loss):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        return {'loss': loss, 'status': status}

    trials = Trials()
    best = fmin(
        objective, space, algo=tpe.suggest, max_evals=max_evals,
        trials=trials, rstate=np.random.RandomState(0)
    )

    return space_eval(space, best), trials