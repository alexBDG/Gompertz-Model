# System modules
import time
import random
import numpy as np

# Figures modules
from opti.utils import get_data
from opti.utils import Gompertz
from opti.utils import HistoryManager
from opti.utils import MultiHistoryManager
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
    shift_min, shift_max = [0., 200.]

    K = 760
    ALPHA = 0.036
    SHIFT = 50
    N_EPOCHS = 1000
    T_BOUNDARY = 0.
    F_BOUNDARY = 16
    N_TRAINING_SAMPLES = 32
    DOMAIN_PHY = (0, 300)
    DOMAIN_TRAIN = (0, 200)

    # Initialize data
    t_train, f_train, t_phy, f_phy = get_data(
        DOMAIN_TRAIN, DOMAIN_PHY,
        k=K, alpha=ALPHA, shift=SHIFT, f_0=F_BOUNDARY,
        n_samples=N_TRAINING_SAMPLES
    )

    # Get space
    space = get_search_space(
        k_min=k_min, k_max=k_max,
        alpha_min=alpha_min, alpha_max=alpha_max,
        shift_min=shift_min, shift_max=shift_max,
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


    history = MultiHistoryManager(N_TRAINING_SAMPLES, F_BOUNDARY, t_phy)
    for k in range(N_TRAINING_SAMPLES):
        print(f"[INFO] N_DATA: {t_train[:k+1].shape[0]}/{N_TRAINING_SAMPLES}")
        # Without Physical Loss
        history_k = HistoryManager(N_EPOCHS, F_BOUNDARY, t_phy)
        args, trials = optimize(
            t_train[:k+1], f_train[:k+1], t_phy,
            space=space,
            f_0=F_BOUNDARY,
            history=history_k,
            add_physical_loss=False,
            max_evals=N_EPOCHS,
        )
        best_iteration = np.argmin(history_k.loss)
        # With Physical Loss
        history_k_sec = HistoryManager(N_EPOCHS, F_BOUNDARY, t_phy)
        args_sec, trials = optimize(
            t_train[:k+1], f_train[:k+1], t_phy,
            space=space,
            f_0=F_BOUNDARY,
            history=history_k_sec,
            add_physical_loss=True,
            max_evals=N_EPOCHS,
        )
        best_iteration_sec = np.argmin(history_k.loss)
        # Trace
        history.update(
            loss=history_k.loss[best_iteration],
            loss_data=history_k.loss_data[best_iteration],
            loss_phy=history_k.loss_phy[best_iteration],
            k=args.get("k"),
            alpha=args.get("alpha"),
            shift=args.get("shift"),
            loss_sec=history_k_sec.loss[best_iteration_sec],
            loss_data_sec=history_k_sec.loss_data[best_iteration_sec],
            loss_phy_sec=history_k_sec.loss_phy[best_iteration_sec],
            k_sec=args_sec.get("k"),
            alpha_sec=args_sec.get("alpha"),
            shift_sec=args_sec.get("shift"),
        )

    # Display results
    history.plot_learning_iterations()
    # history.plot_best_iteration(
    #     t_train, f_train, f_phy, k_true=K, alpha_true=ALPHA, **args
    # )
    history.plot_dynamic_evolution(t_train, f_train, f_phy)
