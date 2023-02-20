"""Module that contains all utility functions."""

import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
plt.rcParams["animation.html"] = "jshtml"


# Set seed
random.seed(0)
np.random.seed(0)



class Gompertz:
    """Create a configured Gompertz function.

    Parameters
    ----------
    k : float
        Limit capacity of the medium.
    alpha : float
        Constant of the function.
    f_0 : float
        Initial term `$f(t=0) = f_0$`.
    """
    def __init__(self, k: float, alpha: float, f_0: float):
        self.k = k
        self.alpha = alpha
        self.f_0 = f_0
        # Compute beta variable
        self.beta = np.log(self.f_0 / self.k)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Analytic solution of the Gompertz function.

        Parameters
        ----------
        t : np.ndarray
            Input of the function, shape of (n,)

        Returns
        -------
        np.ndarray
            Gompertz model evaluated to the `t` input, shape of (n,)
        """
        return self.k * np.exp(self.beta * np.exp(-self.alpha * t))


class HistoryManager:
    def __init__(self, n_epochs, f_0, t_phy):
        self.loss = np.zeros((n_epochs,))
        self.loss_data = np.zeros((n_epochs,))
        self.loss_phy = np.zeros((n_epochs,))
        self.lr = np.zeros((n_epochs,))
        self.k = np.zeros((n_epochs,))
        self.alpha = np.zeros((n_epochs,))
        self.epoch = 0
        # Gompertz
        self.t_phy = t_phy
        self.f_0 = f_0
        self.gompertz = np.zeros((n_epochs//100, t_phy.shape[0]))
        self.prediction = np.zeros((n_epochs//100, t_phy.shape[0]))

    def update(self, loss, loss_data, loss_phy, k, alpha, f_phy_pred):
        self.loss[self.epoch] = loss
        self.loss_data[self.epoch] = loss_data
        self.loss_phy[self.epoch] = loss_phy
        self.k[self.epoch] = k
        self.alpha[self.epoch] = alpha
        if self.epoch % 100:
            self.update_gompertz()
            self.update_predictions(f_phy_pred)
        self.epoch += 1

    def update_gompertz(self):
        # Gompertz
        f_nn = Gompertz(
            k=self.k[self.epoch], alpha=self.alpha[self.epoch], f_0=self.f_0
        )
        self.gompertz[self.epoch//100] = f_nn(self.t_phy)

    def update_predictions(self, f_phy_pred):
        # Gompertz
        self.prediction[self.epoch//100] = f_phy_pred

    def plot_learning_iterations(self):
        fig, axs = plt.subplots(2, 3, sharex=True, figsize=(3*6.4, 1.5*4.8))
        # Display data
        axs[0, 0].plot(self.loss)
        axs[1, 0].plot(self.lr, color="blue")
        axs[0, 1].plot(self.loss_data)
        axs[1, 1].plot(self.loss_phy)
        axs[0, 2].plot(self.k)
        axs[1, 2].plot(self.alpha)
        # Set labels
        axs[0, 0].set_title("loss")
        axs[1, 0].set_title("learning rate")
        axs[0, 1].set_title("loss_data")
        axs[1, 1].set_title("loss_phy")
        axs[0, 2].set_title("K")
        axs[1, 2].set_title("alpha")
        # Set view
        for i, j in itertools.product([0, 1], [0, 1]):
            axs[i, j].set_xlabel("epochs")
        for i, j in itertools.product([0, 1], [0, 1]):
            axs[i, j].set_yscale("log")
            axs[i, j].grid("major")
        fig.tight_layout()
        plt.show()

    def plot_dynamic_evolution(self, f_phy, t_train, f_train):
        fig, axs = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))

        def animate(i):
            fig.suptitle(
                f'Iterration: {i*100} - Loss: {self.loss[i]:.1E}', fontsize=16
            )
            for j in range(2):
                axs[j].cla()
            axs[0].plot(self.t_phy, self.gompertz[i], label="Computed")
            axs[1].plot(self.t_phy, self.prediction[i], label="Computed")
            for j in range(2):
                axs[j].plot(self.t_phy, f_phy, label="True solution")
                axs[j].scatter(
                    t_train, f_train, label="Training data", color="red"
                )
                axs[j].set_xlim(self.t_phy[0], self.t_phy[-1])
                axs[j].set_ylim(-np.max(f_phy) * 0.1, np.max(f_phy) * 1.1)
                axs[j].legend()
                axs[j].set_xlabel("t")
                axs[j].set_ylabel("f(t)")
            axs[0].set_title("$Gompertz_{PiNN}$")
            axs[1].set_title("$PiNN$ prediction")

        ani = matplotlib.animation.FuncAnimation(
            fig, animate, frames=self.gompertz.shape[0], repeat_delay=5
        )
        plt.show()

    def plot_best_iteration(self, t_train, f_train, f_phy,
                            k_true, alpha_true, k, alpha):
        f_nn = Gompertz(k=k, alpha=alpha, f_0=self.f_0)

        title = f"\alpha: {alpha:.3f}/{alpha_true:.3f} - K: {k:.2f}/{k_true:.2f}"

        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(6.4, 4.8))
        fig.suptitle(title, fontsize=16)
        # From params
        axs.plot(self.t_phy, f_nn(self.t_phy), label="$Gompertz_{NN}$ solution")
        axs.plot(self.t_phy, f_phy, label="True solution")
        axs.scatter(t_train, f_train, label="Training data", color="red")
        axs.legend()
        axs.set_xlabel("t")
        axs.set_ylabel("f(t)")
        plt.show()
