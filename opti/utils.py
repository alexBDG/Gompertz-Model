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


def check_nan(value, nan=1e9):
    """Replace a np.nan by `nan` default value.

    Parameters
    ----------
    value : float
        Possible NaN number.
    nan : float, default=1e9
        Repace value.

    Returns
    -------
    float
        Cleaned value.
    """
    if np.isnan(value):
        value = nan
    return value


def get_data(domain_train, domain_phy, k, alpha, shift, f_0, n_samples):
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
    shift : float
        Constant determining the rise of the curve.
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

    f_analytical = Gompertz(k=k, f_0=f_0, alpha=alpha, shift=shift)

    f_train = f_analytical(t_train)
    f_train += np.random.normal(scale=k * 0.05, size=n_samples)
    f_phy = f_analytical(t_phy)

    return t_train, f_train, t_phy, f_phy


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
    def __init__(self, k: float, alpha: float, shift: float, f_0: float):
        self.k = k
        self.alpha = alpha
        self.shift = shift
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
        return self.k * np.exp(
            self.beta * np.exp(-self.alpha * (t - self.shift))
        )


class HistoryManager:
    def __init__(self, n_epochs, f_0, t_phy):
        self.loss = np.zeros((n_epochs,))
        self.loss_data = np.zeros((n_epochs,))
        self.loss_phy = np.zeros((n_epochs,))
        self.k = np.zeros((n_epochs,))
        self.alpha = np.zeros((n_epochs,))
        self.shift = np.zeros((n_epochs,))
        self.epoch = 0
        self.saving_step = n_epochs//10 if n_epochs > 100 else 1
        # Gompertz
        self.t_phy = t_phy
        self.f_0 = f_0
        self.gompertz = np.zeros((n_epochs//self.saving_step, t_phy.shape[0]))

    def update(self, loss, loss_data, loss_phy, k, alpha, shift):
        self.loss[self.epoch] = check_nan(loss)
        self.loss_data[self.epoch] = check_nan(loss_data)
        self.loss_phy[self.epoch] = check_nan(loss_phy)
        self.k[self.epoch] = k
        self.alpha[self.epoch] = alpha
        self.shift[self.epoch] = shift
        if self.epoch % self.saving_step == 0:
            # Gompertz
            f_nn = Gompertz(
                k=self.k[self.epoch], alpha=self.alpha[self.epoch],
                shift=self.shift[self.epoch], f_0=self.f_0
            )
            self.gompertz[self.epoch//self.saving_step] = f_nn(self.t_phy)
        self.epoch += 1

    def plot_learning_iterations(self):
        fig, axs = plt.subplots(2, 3, sharex=True, figsize=(3*6.4, 1.5*4.8))
        # Display data
        axs[0, 0].plot(self.loss)
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
        axs[1, 0].set_visible(False)
        fig.tight_layout()
        plt.show()

    def plot_dynamic_evolution(self, t_train, f_train, f_phy):
        fig, axs = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8))

        def animate(i):
            fig.suptitle((
                f"Iterration: {i*self.saving_step} - "
                f"Loss: {self.loss[i]:.1E}"
            ), fontsize=16)
            axs.cla()
            # axs.plot(self.t_phy, self.gompertz[i], label="Computed")
            axs.plot(self.t_phy, self.prediction[i], label="Computed")
            axs.plot(self.t_phy, f_phy, label="True solution")
            axs.scatter(
                t_train[:i+1], f_train[:i+1],
                label="Training data", color="red"
            )
            axs.set_xlim(self.t_phy[0], self.t_phy[-1])
            axs.set_ylim(-np.max(f_phy) * 0.1, np.max(f_phy) * 1.1)
            axs.legend()
            axs.set_xlabel("t")
            axs.set_ylabel("f(t)")
            axs.set_title("$Gompertz_{PiNN}$")

        ani = matplotlib.animation.FuncAnimation(
            fig, animate, frames=self.gompertz.shape[0], repeat_delay=5
        )
        plt.show()

    def plot_best_iteration(self, t_train, f_train, f_phy,
                            k_true, alpha_true, k, alpha):
        f_nn = Gompertz(k=k, alpha=alpha, f_0=self.f_0)

        title = f"{{\alpha}}: {alpha:.3f}/{alpha_true:.3f} - K: {k:.2f}/{k_true:.2f}"

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


class MultiHistoryManager(HistoryManager):
    def __init__(self, n_epochs, f_0, t_phy):
        super().__init__(n_epochs, f_0, t_phy)
        self.loss_sec = np.zeros((n_epochs,))
        self.loss_data_sec = np.zeros((n_epochs,))
        self.loss_phy_sec = np.zeros((n_epochs,))
        self.k_sec = np.zeros((n_epochs,))
        self.alpha_sec = np.zeros((n_epochs,))
        self.shift_sec = np.zeros((n_epochs,))
        # Gompertz
        self.gompertz_sec = np.zeros(
            (n_epochs//self.saving_step, t_phy.shape[0])
        )

    def update(self, loss, loss_data, loss_phy, k, alpha, shift, loss_sec,
               loss_data_sec, loss_phy_sec, k_sec, alpha_sec, shift_sec):
        self.loss_sec[self.epoch] = check_nan(loss_sec)
        self.loss_data_sec[self.epoch] = check_nan(loss_data_sec)
        self.loss_phy_sec[self.epoch] = check_nan(loss_phy_sec)
        self.k_sec[self.epoch] = k_sec
        self.alpha_sec[self.epoch] = alpha_sec
        self.shift_sec[self.epoch] = shift_sec
        if self.epoch % self.saving_step == 0:
            # Gompertz
            f_nn = Gompertz(
                k=self.k_sec[self.epoch], alpha=self.alpha_sec[self.epoch],
                shift=self.shift_sec[self.epoch], f_0=self.f_0
            )
            self.gompertz_sec[self.epoch//self.saving_step] = f_nn(self.t_phy)
        super().update(loss, loss_data, loss_phy, k, alpha, shift)

    def plot_learning_iterations(self):
        fig, axs = plt.subplots(3, 2, sharex=True, figsize=(2*6.4, 3*4.8))
        # Display data
        axs[0, 0].plot(self.loss, color="red", label="No")
        axs[0, 0].plot(self.loss_sec, color="blue", label="Yes")
        axs[1, 0].plot(self.loss_data, color="red", label="No")
        axs[1, 0].plot(self.loss_data_sec, color="blue", label="Yes")
        axs[2, 0].plot(self.loss_phy, color="red", label="No")
        axs[2, 0].plot(self.loss_phy_sec, color="blue", label="Yes")
        axs[0, 1].plot(self.k, color="red", label="No")
        axs[0, 1].plot(self.k_sec, color="blue", label="Yes")
        axs[1, 1].plot(self.alpha, color="red", label="No")
        axs[1, 1].plot(self.alpha_sec, color="blue", label="Yes")
        axs[2, 1].plot(self.shift, color="red", label="No")
        axs[2, 1].plot(self.shift_sec, color="blue", label="Yes")
        # Set labels
        axs[0, 0].set_title("loss")
        axs[1, 0].set_title("loss_data")
        axs[2, 0].set_title("loss_phy")
        axs[0, 1].set_title("K")
        axs[1, 1].set_title("alpha")
        axs[2, 1].set_title("shift")
        # Set view
        for i, j in itertools.product([0, 1, 2], [0, 1]):
            axs[i, j].set_xlabel("epochs")
            axs[i, j].legend(title="Physical Loss")
            if  j == 0:
                axs[i, j].set_yscale("log")
                axs[i, j].grid("major")
        fig.tight_layout()
        plt.show()

    def plot_dynamic_evolution(self, t_train, f_train, f_phy):
        fig, axs = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))

        def animate(i):
            fig.suptitle(f"Iteration: {i*self.saving_step}\n", fontsize=16)
            for k in range(2):
                axs[k].cla()
                axs[k].plot(self.t_phy, f_phy, label="True solution")
                axs[k].scatter(
                    t_train[:i+1], f_train[:i+1],
                    label="Training data", color="red"
                )
                axs[k].set_xlim(self.t_phy[0], self.t_phy[-1])
                axs[k].set_ylim(-np.max(f_phy) * 0.1, np.max(f_phy) * 1.1)
                axs[k].legend()
                axs[k].set_xlabel("t")
                axs[k].set_ylabel("f(t)")
            axs[0].set_title(f"Loss: {self.loss[i]:.1E} (only data)")
            axs[1].set_title(f"Loss: {self.loss_sec[i]:.1E} (with physical)")
            axs[0].plot(self.t_phy, self.gompertz[i], label="Computed")
            axs[1].plot(self.t_phy, self.gompertz_sec[i], label="Computed")

        ani = matplotlib.animation.FuncAnimation(
            fig, animate, frames=self.gompertz.shape[0], repeat_delay=5
        )
        plt.show()
