import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import cauchy

# I will import several functions from the script "kuramoto.py"
sys.path.append(".")


def initialize_oscillators(
    num_oscillators: int,
    distribution: str = "cauchy",
    scale_omega: float = 1.0,
    scale_phase: float = 1.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes the phases and natural frequencies of the oscillators.

    Parameters
    ----------
    num_oscillators : int
        Number of oscillators.
    distribution : str, optional
        Distribution of natural frequencies ('uniform', 'normal' or 'cauchy').
        Kuramoto uses unimodal distributions, such as the normal distribution.
    scale_omega : float, optional
        Standard deviation of the normal distribution, by default 1.0.
    scale_phase : float, optional
        Scale of the initial phases, by default 1.0.
    seed : int, optional
        Seed for the random number generator, by default None.
    Returns
    -------
    theta : ndarray
        Initial phases of the oscillators.
    omega : ndarray
        Natural frequencies of the oscillators.
    """
    # Set the seed for reproducibility  (optional)
    np.random.seed(seed)

    # Assign a random initial phase to each oscillator
    # (position in the unit circle)
    theta = np.random.uniform(0, 2 * np.pi * scale_phase, num_oscillators)

    # Assign a random natural frequency to each oscillator (angular velocity)
    if distribution == "uniform":
        omega = np.random.uniform(-1.0, 1.0, num_oscillators)
    elif distribution == "normal":
        omega = np.random.normal(0, scale_omega, num_oscillators)
    elif distribution == "cauchy":
        omega = cauchy.rvs(loc=0, scale=scale_omega, size=num_oscillators)
    else:
        raise ValueError("Distribution must be 'uniform' or 'normal'.")

    return theta, omega


def kuramoto_order_parameter(theta: np.ndarray) -> tuple:
    """
    Computes the order parameter of the Kuramoto model.

    Parameters
    ----------
    theta : np.ndarray
        Phases of the oscillators, in radians. Shape is (N, T).

    Returns
    -------
    r : float
        Order parameter (synchronization index).
    phi : float
        Phase of the order parameter.
    rcosphi : float
        Real part of the order parameter, r * cos(phi).
    rsinphi : float
        Imaginary part of the order parameter, r * sin(phi).
    """
    # Compute the order parameter as r * exp(i * phi)
    order_param = np.mean(np.exp(1j * theta), axis=0)
    # The absolute value of the order parameter is the synchronization index
    r = np.abs(order_param)
    # The angle of the order parameter is the phase of the synchronization
    phi = np.angle(order_param)
    # The real part of the order parameter is r * cos(phi)
    rcosphi = np.real(order_param)
    # The imaginary part of the order parameter is r * sin(phi)
    rsinphi = np.imag(order_param)
    return r, phi, rcosphi, rsinphi


def kuramoto_ode_meanfield(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray = None,
    coupling_strength: float = 1.0,
) -> np.ndarray:
    """
    Computes the time derivative of the phase for each oscillator in the
    Kuramoto model. Uses the mean-field approximation: the coupling term is
    the sine of the difference between the phase centroid and
    the phase of each oscillator.

    Reference: https://en.wikipedia.org/wiki/Kuramoto_model


    Parameters
    ----------
    t : float
        Time (not used in the Kuramoto model).
    theta : np.ndarray
        Phases of the oscillators, in radians.
    omega : np.ndarray
        Natural frequencies of the oscillators.
    coupling_strength : float
        Coupling strength (K), which determines the strength of synchronization.

    Returns
    -------
    np.ndarray
        Time derivative of the phase for each oscillator.
    """
    # Ensure omega is an array and matches the shape of theta
    if omega is None:
        omega = np.ones_like(theta)
    # Keep theta within [0, 2 * pi]
    theta = np.mod(theta, 2 * np.pi)
    # Compute the order parameter
    r, phi, _, _ = kuramoto_order_parameter(theta)
    # Compute the coupling term
    coupling_term = coupling_strength * r * np.sin(phi - theta)
    # Compute the time derivative
    dtheta_dt = omega + coupling_term
    return dtheta_dt


def kuramoto_critical_coupling(
    k: np.ndarray, scale: float = 1.0, distribution: str = "cauchy"
) -> np.ndarray:
    """
    Compute the theoretical order parameter for the Kuramoto model
    given the coupling strength k.

    Parameters
    ----------
    k : numpy.ndarray
        Coupling strength.
    scale : float, optional
        Standard deviation of the Gaussian distribution of the
        natural frequencies, default is 1.0.
    distribution : str, optional
        Distribution of the natural frequencies, default is "cauchy".

    Returns
    -------
    r : numpy.ndarray
        Order parameter.
    """
    # The probability density function g(omega) is given by the Gaussian
    # distribution, and thus g(0) is
    if distribution == "cauchy":
        g0 = cauchy.pdf(0, loc=0, scale=scale)
    elif distribution == "normal":
        g0 = 1 / (scale * np.sqrt(2 * np.pi))
        g20 = -g0 / scale**2  # Second derivative of the Gaussian distribution
    else:
        raise ValueError(f"Invalid distribution {distribution}")
    # Critical coupling strength
    kc = 2.0 / (g0 * np.pi)
    # Theoretical order parameter
    r = np.zeros_like(k)
    if distribution == "cauchy":
        # This formula is only valid when using Cauchy distribution
        r[k >= kc] = np.sqrt(1 - (kc / k[k > kc]))
    else:
        # Approximation for the Gaussian distribution
        mu = (k[k >= kc] - kc) / kc
        r[k >= kc] = np.sqrt((16 / (np.pi * kc**3)) * (mu / (-g20)))
        # Only works near onset! For large k, r = 1
        r = np.minimum(r, 1)
    return r


def draw_kuramoto_diagram(
    num_oscillators: int = 5000,
    distribution: str = "cauchy",
    scale: float = 1.0,
    dt: float = 0.01,
    t_end: float = 100.0,
    kmin: float = 0.0,
    kmax: float = 5.0,
    knum: int = 50,
    seed: int = 1,
):
    """
    Draw the Kuramoto diagram, showing the order parameter as a function
    of the coupling strength. Theoretical and empirical order parameters
    are plotted.

    Parameters
    ----------
    num_oscillators : int, optional
        Number of oscillators, default is 1000.
    distribution : str, optional
        Distribution of the natural frequencies, default is "cauchy".
    scale : float, optional
        Standard deviation of the Gaussian distribution of the
        natural frequencies, default is 0.01.
    dt : float, optional
        Time step for the numerical integration, default is 0.01.
    t_end : float, optional
        End time for the numerical integration, default is 100.0.
    kmin : float, optional
        Minimum coupling strength, default is 0.0.
    kmax : float, optional
        Maximum coupling strength, default is 5.0.
    knum : int, optional
        Number of coupling strengths, default is 50.
    seed : int, optional
        Seed for the random number generator, default is 1.
    """
    # Time span and time points relevant for the numerical integration
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)
    # We will take the last X% of the time points to compute the order parameter
    idx_end = int(len(t_eval) * 0.25)
    t_eval = t_eval[-idx_end:]
    # Initialize the coupling strength and the empirical order parameter lists
    ls_k = np.linspace(kmin, kmax, knum)
    ls_r_q10 = np.zeros_like(ls_k)
    ls_r_q50 = np.zeros_like(ls_k)
    ls_r_q90 = np.zeros_like(ls_k)

    # Theoretical order parameter
    r_theoretical = kuramoto_critical_coupling(
        ls_k, scale=scale, distribution=distribution
    )

    # Initialize the oscillators
    theta, omega = initialize_oscillators(
        num_oscillators, distribution=distribution, scale_omega=scale, seed=seed
    )

    # Empirical order parameter
    for idx, coupling_strength in enumerate(ls_k):
        sol = solve_ivp(
            kuramoto_ode_meanfield,
            t_span,
            theta,
            t_eval=t_eval,
            args=(omega, coupling_strength),
        )
        theta = sol.y
        # Keep theta within [0, 2 * pi]
        theta = np.mod(theta, 2 * np.pi)

        # Compute the order parameter
        r, phi, rcosphi, rsinphi = kuramoto_order_parameter(theta)

        # Append the mean order parameter of the last X% of the time points
        ls_r_q10[idx] = np.percentile(r, 10)
        ls_r_q50[idx] = np.percentile(r, 50)
        ls_r_q90[idx] = np.percentile(r, 90)

        print(
            f"K = {coupling_strength:.2f}, r (theory) = {r_theoretical[idx]:.2f}"
            f", r (empirical) = {ls_r_q50[idx]:.2f}"
        )

        # Take the last state as the initial condition for the next iteration
        theta = theta[:, -1]

    # Plot the order parameter as a function of time
    fig, ax = plt.subplots()
    ax.plot(ls_k, r_theoretical, label="Theoretical", color="blue")
    # Plot the empirical order parameter as points with error bars
    ax.errorbar(
        ls_k,
        ls_r_q50,
        yerr=[ls_r_q50 - ls_r_q10, ls_r_q90 - ls_r_q50],
        fmt="o",
        label="Empirical",
        color="red",
    )
    ax.set_xlabel("Coupling strength (K)")
    ax.set_ylabel("Order parameter (r)")
    ax.set_title("Kuramoto model")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_kuramoto_diagram()
