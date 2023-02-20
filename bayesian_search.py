# System modules
import time
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
from hyperopt.pyll import scope

# Figures modules
from opti.utils import Gompertz
from opti.utils import HistoryManager
from opti.display import main_plot_vars
from opti.display import main_plot_history
from opti.display import main_plot_histogram


# Set seed
random.seed(0)
np.random.seed(0)



def get_data(domain_train, domain_phy, k, alpha, f_0, n_samples):
    """Generate training and physical data.
    
    Parameters
    ----------
    domain_train : tuple
        Minimum and maximum value of the training domain.
    domain_phy : tuple
        Minimum and maximum value of the physical domain.
    k : float
        Limit capacity of the medium.
    alpha : float
        Constant of the function.
    f_0 : float
        Initial term `$f(t=0) = f_0$`.
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    t_train : ndarray
        Training time data.
    f_train : ndarray
        Training Gompertz data.
    t_phy : ndarray
        Physical time data.
    f_phy : ndarray
        Physical Gompertz data.
    """
    t_train = np.sort(
        (domain_train[1] - domain_train[0]) * \
            np.random.random_sample(n_samples) + domain_train[0],
        axis=0
    )
    t_phy = np.linspace(domain_phy[0], domain_phy[1], num=1000)

    f_analytical = Gompertz(k=k, f_0=f_0, alpha=alpha)

    f_train = f_analytical(t_train)
    f_phy = f_analytical(t_phy)

    return t_train, f_train, t_phy, f_phy


def get_search_space(k_min=0, k_max=1000, alpha_min=0., alpha_max=1.):
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
    >>> space = {'arg2': scope.int(hp.uniform('arg2', 1e1, 1e3))}
    >>>
    >>> # Choose in log normal distribution
    >>> space = {'arg3': hp.lognormal('arg3', 1e-2, 1)}
    >>>
    >>> # Choose in log uniform distribution
    >>> space = {'arg4': hp.loguniform('arg4', np.log(1e-5), np.log(1e-1))}
    """

    space = {
        "k": hp.uniform("k", k_min, k_max),
        "alpha": hp.loguniform("alpha", np.log(alpha_min), np.log(alpha_max))
    }

    return space


def optimize(t_train, f_train, t_phy, space, f_0, history, max_evals=100):

    def objective(args, f_0=f_0):
        k = args.get("k")
        alpha = args.get("alpha")

        f = Gompertz(k=k, alpha=alpha, f_0=f_0)
        f_train_pred = f(t_train)
        f_phy_pred = f(t_phy)
        dfdt_phy_pred = np.gradient(t_phy, t_phy[1]-t_phy[0])

        # Main part - evaluation
        loss_data = np.mean(np.square(
            f_train_pred - f_train
        ))
        loss_phy = np.mean(np.square(
            dfdt_phy_pred - alpha * f_phy_pred * np.log(k / f_phy_pred)
        ))
        loss = loss_data + loss_phy

        if np.isnan(loss):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        history.update(loss, loss_data, loss_phy, k, alpha, f_phy_pred)

        return {'loss': loss, 'status': status}

    trials = Trials()
    best = fmin(
        objective, space, algo=tpe.suggest, max_evals=max_evals,
        trials=trials, rstate=np.random.RandomState(0)
    )

    return space_eval(space, best), trials


if __name__ == "__main__":
    # Limits
    k_min, k_max = [0, 1e3]
    alpha_min, alpha_max = [1e-3, 1.]

    # Number of iterrations
    max_evals = 2e4

    K = 760
    ALPHA = 0.036
    N_EPOCHS = 1000
    T_BOUNDARY = 0.
    F_BOUNDARY = 16
    N_TRAINING_SAMPLES = 32
    DOMAIN_PHY = (0, 200)
    DOMAIN_TRAIN = (0, 100)

    # Initialize data
    t_train, f_train, t_phy, f_phy = get_data(
        DOMAIN_TRAIN, DOMAIN_PHY,
        k=K, alpha=ALPHA, f_0=F_BOUNDARY, n_samples=N_TRAINING_SAMPLES
    )

    # Initialize history tracer
    history = HistoryManager(N_EPOCHS, F_BOUNDARY, t_phy)

    # Get space
    space = get_search_space(
        k_min=k_min, k_max=k_max,
        alpha_min=alpha_min, alpha_max=alpha_max
    )

    # Logs
    print(f"{'#'*80}\n#{'Starting':^78}#\n{'#'*80}")
    start = time.time()

    # Run
    args, trials = optimize(
        t_train, f_train, t_phy,
        space=space,
        f_0=F_BOUNDARY,
        history=history,
        max_evals=N_EPOCHS,
    )

    # Display results
    main_plot_history(trials=trials)
    main_plot_histogram(trials=trials)
    main_plot_vars(
        trials=trials,
        columns=3,
        arrange_by_loss=False,
        space=space
    )
    history.plot_learning_iterations()
    history.plot_best_iteration(
        t_train, f_train, f_phy, k_true=K, alpha_true=ALPHA, **args
    )
    # history.plot_dynamic_evolution(t_train, f_train, f_phy)
