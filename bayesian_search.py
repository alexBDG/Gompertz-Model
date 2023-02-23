# System modules
import time
import random
import numpy as np

# Figures modules
from opti.utils import get_data
from opti.utils import Gompertz
from opti.utils import HistoryManager
from opti.display import main_plot_vars
from opti.display import main_plot_history
from opti.display import main_plot_histogram
from opti.optimization import optimize
from opti.optimization import get_search_space


# Set seed
random.seed(0)
np.random.seed(0)



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

    # Get space
    space = get_search_space(
        k_min=k_min, k_max=k_max,
        alpha_min=alpha_min, alpha_max=alpha_max
    )

    # Logs
    print(f"{'#'*80}\n#{'Starting':^78}#\n{'#'*80}")
    start = time.time()

    # # Initialize history tracer
    # history = HistoryManager(N_EPOCHS, F_BOUNDARY, t_phy)

    # # Run
    # args, trials = optimize(
    #     t_train, f_train, t_phy,
    #     space=space,
    #     f_0=F_BOUNDARY,
    #     history=history,
    #     max_evals=N_EPOCHS,
    # )

    # # Display results
    # main_plot_history(trials=trials)
    # main_plot_histogram(trials=trials)
    # main_plot_vars(
    #     trials=trials,
    #     columns=3,
    #     arrange_by_loss=False,
    #     space=space
    # )
    # history.plot_learning_iterations()
    # history.plot_best_iteration(
    #     t_train, f_train, f_phy, k_true=K, alpha_true=ALPHA, **args
    # )
    # # history.plot_dynamic_evolution(t_train, f_train, f_phy)


    history = HistoryManager(N_TRAINING_SAMPLES, F_BOUNDARY, t_phy)
    for k in range(N_TRAINING_SAMPLES):
        print(f"[INFO] N_DATA: {t_train[:k+1].shape[0]}/{N_TRAINING_SAMPLES}")
        history_k = HistoryManager(100, F_BOUNDARY, t_phy)
        args, trials = optimize(
            t_train[:k+1], f_train[:k+1], t_phy,
            space=space,
            f_0=F_BOUNDARY,
            history=history_k,
            max_evals=100,
        )
        best_iteration = np.argmin(history.loss)
        history.update(
            loss=history_k.loss[best_iteration],
            loss_data=history.loss_data[best_iteration],
            loss_phy=history.loss_phy[best_iteration],
            f_phy_pred=Gompertz(f_0=F_BOUNDARY, **args)(t_phy),
            **args,
        )

    # Display results
    history.plot_learning_iterations()
    # history.plot_best_iteration(
    #     t_train, f_train, f_phy, k_true=K, alpha_true=ALPHA, **args
    # )
    history.plot_dynamic_evolution(t_train, f_train, f_phy)
