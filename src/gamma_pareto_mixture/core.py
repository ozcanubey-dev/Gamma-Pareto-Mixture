"""
Gamma-Pareto Distribution Package

A Python implementation of the Gamma-Pareto mixture distribution for modelling
heavy-tailed data. This module provides statistical functions, parameter estimation,
simulation, and visualization tools for the Gamma-Pareto distribution.

Author: Ubeydullah Ozcan
License: MIT
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.special import gamma, digamma, gammainc
from scipy.optimize import minimize, fsolve
from scipy import stats
from typing import Union, List, Dict, Tuple, Optional, Any


def simulate_gamma_pareto_mixture(
    n_sim: int,
    alpha: float,
    c: float,
    theta: float,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate random samples from a Gamma-Pareto distribution.

    Using Lemma 1: If Y ~ Gamma(α, c), then X = θ * exp(Y) ~ Gamma-Pareto(α, c, θ)

    Parameters
    ----------
    n_sim : int
        Number of samples to generate. Must be positive.
    alpha : float
        Shape parameter (α > 0).
    c : float
        Scale parameter (c > 0).
    theta : float
        Minimum value parameter (θ > 0).
    random_state : int, optional
        Random seed for reproducibility. If None, uses current random state.

    Returns
    -------
    np.ndarray
        Array of generated samples from Gamma-Pareto distribution.

    Raises
    ------
    ValueError
        If n_sim <= 0 or any parameter is non-positive.

    Examples
    --------
    >>> samples = simulate_gamma_pareto_mixture(
    ...     n_sim=1000, alpha=2.5, c=0.3, theta=10.0, random_state=42
    ... )
    >>> len(samples)
    1000
    >>> samples.min() > 10.0
    True
    """
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if c <= 0:
        raise ValueError("c must be > 0")
    if theta <= 0:
        raise ValueError("theta must be > 0")

    if random_state is not None:
        np.random.seed(random_state)

    # Generate from Gamma distribution
    y = np.random.gamma(shape=alpha, scale=c, size=n_sim)

    # Transform to Gamma-Pareto
    x = theta * np.exp(y)

    return x


def ks_test_gpm(
    observed: Union[List[float], np.ndarray],
    theoretical: Union[List[float], np.ndarray]
) -> Dict[str, Any]:
    """
    Perform Kolmogorov-Smirnov test between observed and theoretical distributions.

    Parameters
    ----------
    observed : array-like
        Observed data values.
    theoretical : array-like
        Theoretical distribution values or samples from reference distribution.

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic' : float
            KS test statistic D
        - 'pvalue' : float
            Two-tailed p-value
        - 'message' : str
            Interpretation of the test result at α=0.05 significance level

    Examples
    --------
    >>> import numpy as np
    >>> observed = np.random.gamma(2, 1, 100)
    >>> theoretical = np.random.gamma(2, 1, 100)
    >>> result = ks_test_gpm(observed, theoretical)
    >>> 'statistic' in result
    True
    >>> 'pvalue' in result
    True
    """
    ks = stats.kstest(observed, theoretical)
    if ks.pvalue > 0.05:
        message = "Null Hypothesis not rejected ⟹ samples are drawn from the same distribution"
    else:
        message = "Null Hypothesis rejected ⟹ samples are NOT drawn from the same distribution"

    return {
        "statistic": ks.statistic,
        "pvalue": ks.pvalue,
        "message": message
    }


class GammaParetoMixture:
    """
    Gamma-Pareto mixture distribution class.

    Implements the statistical properties and estimation methods for the
    Gamma-Pareto distribution as described in statistical literature. The
    distribution is defined by three parameters: shape (α), scale (c), and
    minimum value (θ).

    Parameters
    ----------
    data : Union[pd.DataFrame, pl.DataFrame]
        DataFrame containing the data to analyze. Supports both Pandas and Polars.
    colname : str
        Name of the column containing the data values.

    Attributes
    ----------
    data : DataFrame
        Input data
    colname : str
        Column name containing data
    _data_vector : np.ndarray, optional
        Cached numpy array of data values

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'values': np.random.lognormal(2, 1, 100)})
    >>> gp = GammaParetoMixture(data=data, colname='values')
    >>> result = gp.fit(method='hybrid')
    >>> 'alpha' in result
    True
    >>> 'c' in result
    True
    """

    def __init__(self, data: Union[pd.DataFrame, pl.DataFrame], colname: str) -> None:
        """
        Initialize GammaParetoMixture distribution with data.

        Parameters
        ----------
        data : Union[pd.DataFrame, pl.DataFrame]
            DataFrame containing the data.
        colname : str
            Name of the column to analyze.

        Raises
        ------
        TypeError
            If data is not a Pandas or Polars DataFrame.
        ValueError
            If colname is not in the DataFrame.
        """
        self.data = data
        self.colname = colname
        self._data_vector = None

        # Validate input
        if colname not in data.columns:
            raise ValueError(f"Column '{colname}' not found in DataFrame")

    def get_datatype(self) -> str:
        """
        Get the type of DataFrame (Pandas or Polars).

        Returns
        -------
        str
            'Pandas' for pandas.DataFrame, 'Polars' for polars.DataFrame

        Raises
        ------
        TypeError
            If data is neither Pandas nor Polars DataFrame.
        """
        if isinstance(self.data, pd.DataFrame):
            return "Pandas"
        elif isinstance(self.data, pl.DataFrame):
            return "Polars"
        else:
            raise TypeError("Data must be either pandas.DataFrame or polars.DataFrame")

    def get_data_vector(self) -> np.ndarray:
        """
        Extract data as numpy array.

        Returns
        -------
        np.ndarray
            Numpy array of data values.

        Notes
        -----
        Caches the result to avoid repeated conversion.
        """
        if self._data_vector is not None:
            return self._data_vector

        df_type = self.get_datatype()
        if df_type == "Pandas":
            data_vector = self.data[self.colname].to_numpy()
        else:
            data_vector = self.data.get_column(self.colname).to_numpy()

        self._data_vector = data_vector
        return data_vector

    def pdf(
        self,
        alpha: float,
        c: float,
        theta: float
    ) -> Dict[str, np.ndarray]:
        """
        Probability density function of Gamma-Pareto distribution.

        The PDF is defined as:
        g(x) = 1/(x·Γ(α)·c^α) · (θ/x)^(1/c) · [log(x/θ)]^(α-1)

        Parameters
        ----------
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'x': np.ndarray - Grid points for PDF evaluation
            - 'pdf': np.ndarray - PDF values at grid points

        Examples
        --------
        >>> gp = GammaParetoMixture(data, 'values')
        >>> pdf_result = gp.pdf(alpha=2.0, c=0.5, theta=10.0)
        >>> len(pdf_result['x'])
        1000
        >>> len(pdf_result['pdf'])
        1000
        """
        data = self.get_data_vector()
        x = np.linspace(np.min(data), np.max(data), 1000)
        mask = x > theta
        result = np.zeros_like(x, dtype=float)

        if np.any(mask):
            x_masked = x[mask]
            term1 = 1 / (x_masked * gamma(alpha) * c ** alpha)
            term2 = (theta / x_masked) ** (1 / c)
            term3 = np.log(x_masked / theta) ** (alpha - 1)
            result[mask] = term1 * term2 * term3

        return {'x': x, 'pdf': result}

    def cdf(
        self,
        alpha: float,
        c: float,
        theta: float
    ) -> Dict[str, np.ndarray]:
        """
        Cumulative distribution function of Gamma-Pareto distribution.

        From equation (2.3):
        G(x) = γ(α, (1/c)·log(x/θ)) / Γ(α)
        where γ(α, t) is the lower incomplete gamma function.

        Parameters
        ----------
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'x': np.ndarray - Sorted data values
            - 'cdf': np.ndarray - CDF values at data points

        Examples
        --------
        >>> cdf_result = gp.cdf(alpha=2.0, c=0.5, theta=10.0)
        >>> cdf_result['cdf'][-1]  # Last CDF value should be ~1
        1.0
        """
        x = np.sort(self.get_data_vector())
        mask = x > theta

        result = np.zeros_like(x, dtype=float)
        result[x <= theta] = 0

        if np.any(mask):
            x_masked = x[mask]
            # Using gammainc which is the regularized incomplete gamma function
            # gammainc(a, x) = γ(a, x) / Γ(a)
            result[mask] = gammainc(alpha, (1 / c) * np.log(x_masked / theta))

        return {'x': x, 'cdf': result}

    def hazard_fct(
        self,
        alpha: float,
        c: float,
        theta: float
    ) -> Dict[str, np.ndarray]:
        """
        Hazard (failure rate) function of Gamma-Pareto distribution.

        h(x) = g(x) / [1 - G(x)]

        Parameters
        ----------
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'x': np.ndarray - Grid points for hazard function
            - 'hazard': np.ndarray - Hazard values

        Examples
        --------
        >>> hazard_result = gp.hazard_fct(alpha=2.0, c=0.5, theta=10.0)
        >>> hazard_result['hazard'].min() >= 0
        True
        """
        data = self.get_data_vector()
        x = np.linspace(np.min(data), np.max(data), 1000)
        mask = x > theta

        result = np.zeros_like(x, dtype=float)
        result[x <= theta] = 0

        if np.any(mask):
            pdf_vals = self.pdf(alpha, c, theta)['pdf']
            cdf_vals = self.cdf(alpha, c, theta)['cdf']
            result[mask] = pdf_vals[mask] / (1 - cdf_vals[mask])

        return {'x': x, 'hazard': result}

    def moments(
        self,
        r: int,
        alpha: float,
        c: float,
        theta: float
    ) -> Dict[str, Any]:
        """
        Compute the r-th non-central moment of Gamma-Pareto distribution.

        From equation (4.2):
        E(X^r) = θ^r · (1 - r·c)^(-α), for c < 1/r

        Parameters
        ----------
        r : int
            Order of moment (must be positive)
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'Order': int - Order of moment
            - 'value': float - Moment value (inf if condition not met)
            - 'message': str - Condition check result

        Examples
        --------
        >>> moment = gp.moments(1, alpha=2.0, c=0.3, theta=10.0)
        >>> moment['Order']
        1
        >>> isinstance(moment['value'], float)
        True
        """
        if r <= 0:
            raise ValueError("Moment order r must be positive")

        if r * c >= 1:
            return {
                "Order": r,
                "value": np.inf,
                "message": "c < 1/r condition is not met!"
            }
        return {
            "Order": r,
            "value": theta ** r * (1 - r * c) ** (-alpha),
            "message": "c < 1/r condition is met!"
        }

    def mean(
        self,
        alpha: float,
        c: float,
        theta: float
    ) -> Dict[str, float]:
        """
        Mean (first moment) of Gamma-Pareto distribution.

        μ = E(X) = θ · (1 - c)^(-α), for c < 1

        Parameters
        ----------
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        dict
            Dictionary with key 'Mean' containing the mean value.

        Examples
        --------
        >>> mean_result = gp.mean(alpha=2.0, c=0.3, theta=10.0)
        >>> mean_result['Mean'] > 10.0
        True
        """
        mu = self.moments(1, alpha, c, theta)
        return {"Mean": mu["value"]}

    def variance(
        self,
        alpha: float,
        c: float,
        theta: float
    ) -> Dict[str, Any]:
        """
        Variance of Gamma-Pareto distribution.

        Var(X) = E(X²) - [E(X)]²
        Requires c < 1/2 for finite variance.

        Parameters
        ----------
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'Variance': float - Variance value (inf if condition not met)
            - 'message': str - Condition check result

        Examples
        --------
        >>> var_result = gp.variance(alpha=2.0, c=0.3, theta=10.0)
        >>> var_result['Variance'] > 0
        True
        """
        if 2 * c >= 1:
            return {
                "Variance": np.inf,
                "message": "c < 1/2 condition is not met!"
            }
        moment1 = self.moments(1, alpha, c, theta)
        moment2 = self.moments(2, alpha, c, theta)
        var = moment2['value'] - moment1['value'] ** 2
        return {
            "Variance": var,
            "message": "c < 1/2 condition is met!"
        }

    def mode(
        self,
        alpha: float,
        c: float,
        theta: float
    ) -> Dict[str, Any]:
        """
        Mode (most frequent value) of Gamma-Pareto distribution.

        From Theorem 2:
        - If α ≤ 1: mode = θ
        - If α > 1: mode = θ · exp(c·(α-1)/(c+1))

        Parameters
        ----------
        alpha : float
            Shape parameter (α > 0)
        c : float
            Scale parameter (c > 0)
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'Mode': float - Mode value
            - 'message': str - Condition description

        Examples
        --------
        >>> mode_result = gp.mode(alpha=2.0, c=0.5, theta=10.0)
        >>> mode_result['Mode'] > 10.0
        True
        """
        if alpha <= 1:
            return {
                "Mode": theta,
                "message": "alpha ≤ 1, Mode = Theta"
            }
        else:
            return {
                "Mode": theta * np.exp(c * (alpha - 1) / (c + 1)),
                "message": f"Mode = θ·exp(c·(α-1)/(c+1))"
            }

    def neg_loglikelihood(
        self,
        params: List[float],
        theta: float
    ) -> Union[float, Dict[str, Any]]:
        """
        Negative log-likelihood function for Gamma-Pareto distribution.

        From equation (5.1):
        log L(α, c) = Σ [-α·log c - log Γ(α) - log θ - (1 + 1/c)·log(x_i/θ)
                        + (α - 1)·log(log(x_i/θ))]

        Parameters
        ----------
        params : List[float]
            List containing [alpha, c]
        theta : float
            Minimum value parameter (θ > 0)

        Returns
        -------
        Union[float, Dict[str, Any]]
            Negative log-likelihood value, or dictionary with error message
            if parameters invalid.

        Notes
        -----
        Following Smith's method, θ is estimated by sample minimum.
        Returns -inf for α ≤ 0.
        """
        alpha, c = params
        data = self.get_data_vector()

        if alpha <= 0:
            return {
                "Loglikelihood": -np.inf,
                "message": "alpha ≤ 0 and theta is min of sample ⟹ No MLE solution for alpha and c"
            }
        else:
            mask = data > theta
            data_filtered = data[mask]

            if len(data_filtered) == 0:
                return {
                    "Loglikelihood": np.nan,
                    "message": "dataset size is null"
                }
            else:
                # Calculate terms
                log_ratio = np.log(data_filtered / theta)
                log_log_ratio = np.log(log_ratio)

                # Log-likelihood components
                term1 = -alpha * np.log(c)
                term2 = -np.log(gamma(alpha))
                term3 = -np.log(theta)
                term4 = -(1 + 1 / c) * log_ratio
                term5 = (alpha - 1) * log_log_ratio

                log_lik = np.sum(term1 + term2 + term3 + term4 + term5)
                return -log_lik  # Negative for minimization

    def fit(
        self,
        method: str = 'hybrid'
    ) -> Dict[str, Any]:
        """
        Fit Gamma-Pareto distribution to data using Maximum Likelihood Estimation.

        Following the procedure in Section 5:
        1. Estimate θ as sample minimum x_{(1)}
        2. Solve for α and c using MLE equations

        Parameters
        ----------
        method : str, optional
            Estimation method:
            - 'hybrid' (default): Use equation solving for α, then compute c
            - 'optimize': Use numerical optimization

        Returns
        -------
        dict
            Dictionary containing:
            - 'message': str - Estimation method or status
            - 'loglik': float - Maximized log-likelihood
            - 'alpha': float - Estimated shape parameter
            - 'alpha_se': float - Approximate standard error for α
            - 'c': float - Estimated scale parameter
            - 'c_se': float - Approximate standard error for c
            - 'theta': float - Estimated minimum value (sample minimum)
            - 'AIC': float - Akaike Information Criterion

        Raises
        ------
        ValueError
            If insufficient data points above minimum.
        RuntimeError
            If optimization fails.

        Examples
        --------
        >>> result = gp.fit(method='hybrid')
        >>> isinstance(result['alpha'], float)
        True
        >>> isinstance(result['c'], float)
        True
        >>> result['AIC'] > 0
        True
        """
        data = self.get_data_vector()
        theta_hat = np.min(data)

        data_filtered = data[data > np.min(data)]
        n_filtered = len(data_filtered)

        if n_filtered < 2:
            raise ValueError("Insufficient data points above minimum for MLE")

        # Calculate statistics needed for estimation
        log_ratio = np.log(data_filtered / np.min(data))
        m1_star = np.mean(log_ratio)
        log_log_ratio = np.log(log_ratio)
        m2_star = np.mean(log_log_ratio)

        if method == 'hybrid':
            # Method 1: Solve equation (5.6) for α, then compute c from (5.5)
            # Equation (5.6): ψ(α) - log(α) + log(m1*) - m2* = 0

            def equation_for_alpha(alpha):
                return digamma(alpha) - np.log(alpha) + np.log(m1_star) - m2_star

            # Find initial guess for α
            # From the paper: α₀ = ȳ² / s_y²
            y = log_ratio
            y_mean = np.mean(y)
            y_var = np.var(y, ddof=1)
            alpha_init = y_mean ** 2 / y_var if y_var > 0 else 1.0

            # Solve for α
            try:
                alpha_solution = fsolve(equation_for_alpha, alpha_init)[0]
                if alpha_solution > 0:
                    print("Solution for alpha using Hybrid method exists!")
                    c = m1_star / alpha_solution
                    log_lik = self.neg_loglikelihood([alpha_solution, c], theta_hat)

                    # Approximate standard errors
                    alpha_se_approx = np.sqrt(
                        6 * alpha_solution ** 3 / (n_filtered * (3 * alpha_solution + 1))
                    )
                    c_se_approx = np.sqrt(
                        c ** 2 * (6 * alpha_solution ** 2 + 3 * alpha_solution + 1) /
                        (n_filtered * alpha_solution * (3 * alpha_solution + 1))
                    )

                    return {
                        "message": "Solution using Hybrid method",
                        "loglik": -log_lik,
                        "alpha": alpha_solution,
                        "alpha_se": alpha_se_approx,
                        "c": c,
                        "c_se": c_se_approx,
                        "theta": theta_hat,
                        "AIC": 2 * 3 - 2 * -log_lik
                    }
                else:
                    # Fallback to optimization
                    print("Fallback optimization from 'hybrid' to 'optimize'")
                    method = 'optimize'
            except:
                method = 'optimize'

        if method == 'optimize':
            # Method 2: Numerical optimization of log-likelihood
            # Initial values from method of moments
            y = log_ratio
            y_mean = np.mean(y)
            y_var = np.var(y, ddof=1)
            c_init = y_var / y_mean if y_mean > 0 else 0.1
            alpha_init = y_mean ** 2 / y_var if y_var > 0 else 1.0
            initial_params = (alpha_init, c_init)

            # Bounds for parameters
            bounds = [(1e-6, None), (1e-6, None)]

            # Optimize
            result = minimize(
                self.neg_loglikelihood,
                initial_params,
                args=theta_hat,
                bounds=bounds,
                method='L-BFGS-B'
            )

            if result.success:
                # Approximate standard errors
                alpha_se_approx = np.sqrt(
                    6 * result.x[0] ** 3 / (n_filtered * (3 * result.x[0] + 1))
                )
                c_se_approx = np.sqrt(
                    result.x[1] ** 2 * (6 * result.x[0] ** 2 + 3 * result.x[0] + 1) /
                    (n_filtered * result.x[0] * (3 * result.x[0] + 1))
                )
                return {
                    "message": result.message,
                    "loglik": -result.fun,
                    "alpha": result.x[0],
                    "alpha_se": alpha_se_approx,
                    "c": result.x[1],
                    "c_se": c_se_approx,
                    "theta": theta_hat,
                    "AIC": 2 * 3 - 2 * (-result.fun)
                }
            else:
                raise RuntimeError("Optimization failed: " + result.message)

    def plot_fit(
        self,
        alpha: float,
        c: float,
        theta: float,
        bins: int = 20,
        figsize: Tuple[int, int] = (8, 8)
    ) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
        """
        Create diagnostic plots for Gamma-Pareto distribution fit.

        Generates three subplots:
        1. PDF fit with histogram
        2. CDF comparison (empirical vs fitted)
        3. Q-Q plot

        Parameters
        ----------
        alpha : float
            Shape parameter
        c : float
            Scale parameter
        theta : float
            Minimum value parameter
        filename : str
            Base filename for saving the plot (will save as f'plot_result_{filename}.png')
        bins : int, optional
            Number of histogram bins (default: 20)
        figsize : tuple, optional
            Figure size in inches (default: (8, 8))

        Returns
        -------
        tuple
            Tuple of matplotlib axes objects (ax1, ax2, ax3)

        Examples
        --------
        >>> ax1, ax2, ax3 = gp.plot_fit(
        ...     alpha=2.0, c=0.5, theta=10.0,
        ...     filename='my_analysis', bins=30
        ... )
        >>> isinstance(ax1, plt.Axes)
        True
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Plot 1: PDF fit
        ax1 = axes[0]
        ax1.hist(
            self.get_data_vector(),
            bins=bins,
            density=True,
            alpha=0.6,
            label='Data',
            edgecolor='black'
        )

        pdf_data = self.pdf(alpha=alpha, c=c, theta=theta)
        x_vals = pdf_data['x']
        pdf_vals = pdf_data['pdf']
        ax1.plot(x_vals, pdf_vals, 'r-', linewidth=2, label='Gamma-Pareto fit')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Density')
        ax1.set_title('PDF Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: CDF comparison
        ax2 = axes[1]
        sorted_data = np.sort(self.get_data_vector())
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, ecdf, 'b-', alpha=0.7, label='Empirical CDF')

        cdf_data = self.cdf(alpha=alpha, c=c, theta=theta)
        cdf_vals = cdf_data['cdf']
        ax2.plot(sorted_data, cdf_vals, 'r-', linewidth=2, label='Fitted CDF')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('CDF Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Q-Q plot
        ax3 = axes[2]
        ax3.scatter(cdf_vals, ecdf, alpha=0.6)
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax3.set_xlabel('Theoretical Quantiles')
        ax3.set_ylabel('Empirical Quantiles')
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')

        plt.tight_layout()

        return ax1, ax2, ax3


    @classmethod
    def do_example(
        cls,
        dataset: str = "Fatigue"
    ) -> 'GammaParetoMixture':
        """
        Create a GammaPareto instance with built-in example datasets.

        Available datasets:
        - "Fatigue": Fatigue life of 101 specimens
        - "Flood": Annual maximum flood discharges
        - "Tribolium": Tribolium confusum insect counts

        Parameters
        ----------
        dataset : str
            Name of dataset to load. Must be "Fatigue", "Flood", or "Tribolium"

        Returns
        -------
        GammaParetoMixture
            Initialized GammaPareto instance with example data

        Raises
        ------
        ValueError
            If dataset name is not recognized

        Examples
        --------
        >>> gp_fatigue = GammaParetoMixture.do_example('Fatigue')
        >>> gp_flood = GammaParetoMixture.do_example('Flood')
        >>> gp_tribolium = GammaParetoMixture.do_example('Tribolium')
        """
        if dataset == "Fatigue":
            vec = [
                70, 90, 96, 97, 99, 100, 103, 104, 104, 105,
                107, 108, 108, 108, 109, 109, 112, 112, 113, 114,
                114, 114, 116, 119, 120, 120, 120, 121, 121, 123,
                124, 124, 124, 124, 124, 128, 128, 129, 129, 130,
                130, 130, 131, 131, 131, 131, 132, 132, 132, 132,
                133, 134, 134, 134, 134, 134, 136, 137, 138, 138,
                138, 139, 139, 141, 141, 142, 142, 142, 142, 142,
                142, 142, 144, 144, 145, 146, 148, 148, 149, 151,
                151, 152, 155, 156, 157, 157, 157, 158, 159, 159,
                162, 163, 163, 164, 166, 166, 168, 170, 174, 196, 212
            ]
        elif dataset == "Flood":
            vec = np.array([
                1460, 4050, 3570, 2060, 1300, 1390, 1720, 6280, 1360, 7440,
                5320, 1400, 3240, 2710, 4520, 4840, 8320, 13900, 71500, 6250,
                2260, 318, 1330, 970, 1920, 15100, 2870, 20600, 3810, 726,
                7500, 7170, 2000, 829, 17300, 4740, 13400, 2940, 5660
            ])
        elif dataset == "Tribolium":
            x_values = [
                55, 65, 75, 85, 95, 105, 115, 125, 135, 145,
                155, 165, 175, 185, 195, 205, 215, 225, 235, 245
            ]
            frequencies = [
                3, 20, 53, 78, 86, 86, 68, 51, 20, 11,
                6, 4, 7, 5, 1, 2, 0, 1, 1, 1
            ]

            # Expand to individual data points
            tribolium_data = []
            for x, freq in zip(x_values, frequencies):
                tribolium_data.extend([x] * freq)

            vec = tribolium_data
        else:
            raise ValueError('Dataset must be "Fatigue", "Flood", or "Tribolium"')

        df = pd.DataFrame({dataset: vec})
        return cls(data=df, colname=dataset)


