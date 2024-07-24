"""
Implemented redshift models
"""

import numpy as xp

from ..experimental.cosmo_models import CosmoMixin
from ..utils import powerlaw

__all__ = [
    "_Redshift",
    "PowerLawRedshift",
    "MadauDickinsonRedshift",
    "total_four_volume",
]


class _Redshift(CosmoMixin):
    r"""
    This assumes the model is defined as

    .. math::

        p(z | \Lambda) = \frac{1}{(1 + z)} \frac{dVc}{dz} \psi(z | \Lambda).

    Where :math:`\psi(z | \Lambda)` is implemented in :func:`psi_of_z`.

    Parameters
    ----------
    z_max: float
        The maximum redshift allowed, this is also used to normalize the
        probability.
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.

    Attributes
    ----------
    base_variable_names: list
        :math:`\Lambda` - list of astrophysical rate-evolution parameters
        for the model.
    """

    base_variable_names = None

    @property
    def variable_names(self):
        """
        Variable names for the model

        Returns
        -------
        vars: list
            Variable names including astrophysical rate-evolution parameters
            and cosmological parameters
            (:code:`self.base_variable_names + self.cosmology_names`).
        """
        vars = self.cosmology_names.copy()
        if self.base_variable_names is not None:
            vars += self.base_variable_names
        return vars

    def __init__(self, z_max=2.3, cosmo_model="Planck15"):
        super().__init__(cosmo_model=cosmo_model)
        self.z_max = z_max
        self.zs = xp.linspace(1e-6, z_max, 2500)

    def __call__(self, dataset, **kwargs):
        """
        Wrapper to :func:`probability`.
        """
        return self.probability(dataset=dataset, **kwargs)

    def normalisation(self, parameters):
        r"""
        Compute the normalization of the rate-weighted spacetime volume.

        .. math::

            \mathcal{V} = \int_{0}^{z_{\max}} dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Here, :math:`z_{\max}` is :code:`self.z_max`.

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters

        Returns
        -------
        norm: float | array-like:
            Total rate-weighted comoving spacetime volume
        """
        normalisation_data = self.differential_spacetime_volume(
            dict(redshift=self.zs), bounds=True, **parameters
        )
        norm = xp.trapz(normalisation_data, self.zs)
        return norm

    def probability(self, dataset, **parameters):
        """
        Compute the normalized probability of a merger occurring at the
        specified redshift.

        Parameters
        ----------
        dataset: dict[str, array-like]
            Dictionary of sample points that contains :code:`redshift`
            as a key.
        parameters: dict[str, float]
            Dictionary of parameters :math:`\Lambda`.

        Returns
        -------
        array-like
            The normalized probability per sample
        """
        normalisation = self.normalisation(parameters=parameters)
        differential_volume = self.differential_spacetime_volume(
            dataset=dataset, bounds=True, **parameters
        )
        return differential_volume / normalisation

    def psi_of_z(self, redshift, **parameters):
        r"""
        Method encoding the redshift evolution of the merger rate.
        This should be overwritten in child classes.
        By convention this should be normalized such that

        .. math::

            \psi(0 | \Lambda) = 1.

        Parameters
        ----------
        redshift: array-like
            The redshifts at which to evaluate the model.
        parameters: dict[str, float]
            Dictionary of parameters :math:`\Lambda`.

        Returns
        -------
        array-like
            :math:`\psi(z | \Lambda)` for in input redshifts.
        """
        raise NotImplementedError

    def dvc_dz(self, redshift, **parameters):
        r"""

        .. note::

            The units of this differ from
            :code:`wcosmo.differential_comoving_volume` and
            :code:`astropy.cosmology.FlatwCDM.differential_comoving_volume`
            by a factor of :math:`4 \pi`.

        Returns
        -------
        float
            The differential comoving volume in :math:`{\rm Mpc}^3`
        """
        return (
            4
            * xp.pi
            * self.cosmology(parameters).differential_comoving_volume(redshift)
        )

    def differential_spacetime_volume(self, dataset, bounds=False, **parameters):
        r"""
        Compute the differential spacetime volume.

        .. math::

            d\mathcal{V} = \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        dataset: dict
            Dictionary containing entry "redshift"
        parameters: dict
            Dictionary of parameters

        Returns
        -------
        differential_volume: (float, array-like)
            Differential spacetime volume
        """
        psi_of_z = self.psi_of_z(redshift=dataset["redshift"], **parameters)
        differential_volume = psi_of_z / (1 + dataset["redshift"])
        differential_volume *= self.dvc_dz(redshift=dataset["redshift"], **parameters)
        if bounds:
            differential_volume *= dataset["redshift"] <= self.z_max

        return differential_volume


class PowerLawRedshift(_Redshift):
    base_variable_names = ["lamb"]

    def psi_of_z(self, redshift, **parameters):
        r"""
        Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270
        (`arXiv:1805.10270 <https://arxiv.org/abs/1805.10270>`_.

        .. math::

            \psi(z|\gamma, \kappa, z_p) = (1 + z)^\lambda

        Parameters
        ----------
        lamb: float
            The spectral index :math:`\lambda`.
        """
        return (1 + redshift) ** parameters["lamb"]


class MadauDickinsonRedshift(_Redshift):

    base_variable_names = ["gamma", "kappa", "z_peak"]

    def psi_of_z(self, redshift, **parameters):
        r"""
        Redshift model from Fishbach+
        (`arXiv:1805.10270 <https://arxiv.org/abs/1805.10270>`_ Eq. (33))
        See Callister+ (`arXiv:2003.12152 <https://arxiv.org/abs/2003.12152>`_
        Eq. (2)) for the normalisation.

        .. math::

            \psi(z|\gamma, \kappa, z_p) = \frac{(1 + z)^\gamma}{1 + (\frac{1 + z}{1 + z_p})^\kappa}

        Parameters
        ----------
        gamma: float
            Slope of the distribution at low redshift, :math:`\gamma`.
        kappa: float
            Slope of the distribution at high redshift, :math:`\kappa`.
        z_peak: float
            Redshift at which the distribution peaks, :math:`z_p`.
        """
        gamma = parameters["gamma"]
        kappa = parameters["kappa"]
        z_peak = parameters["z_peak"]
        psi_of_z = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** kappa
        )
        psi_of_z *= 1 + (1 + z_peak) ** (-kappa)
        return psi_of_z


def total_four_volume(lamb, analysis_time, max_redshift=2.3):
    r"""
    Calculate the rate-weighted four-volume for a given power-law redshift model.

    .. math::

        \mathcal{V} = T \int_{0}^{z_{\max}} dz \frac{dV_c}{dz} (1 + z)^{\lambda - 1}

    Parameters
    ----------
    lamb: float
        The spectral index, :math:`\Lambda`.
    analysis_time: float
        The total analysis time, :math:`T`.
    max_redshift: float, optional
        The maximum redshift to integrate to, :math:`z_{\max}`, default=2.3.

    Returns
    -------
    total_volume: float
        The rate-weighted four-volume

    Notes
    -----
    This assumes a :code:`Planck15` cosmology.
    """
    from wcosmo.astropy import Planck15
    from wcosmo.utils import disable_units

    disable_units()

    redshifts = xp.linspace(0, max_redshift, 2500)
    psi_of_z = (1 + redshifts) ** lamb
    normalization = 4 * xp.pi / 1e9 * analysis_time
    total_volume = (
        xp.trapz(
            Planck15.differential_comoving_volume(redshifts)
            / (1 + redshifts)
            * psi_of_z,
            redshifts,
        )
        * normalization
    )
    return total_volume


class TimeDelayRedshift(_Redshift):

    base_variable_names = ["kappa_d", "tau_min"]
    
    def time_delay(self, tau, **parameters):
        kappa_d = parameters["kappa_d"]
        tau_min = parameters["tau_min"]
        
        # if time delay is larger than age of universe, no merger would've happened yet
        hubble_time = self.cosmology(parameters).hubble_time
        
        return powerlaw(tau, kappa_d, hubble_time, tau_min)
            
    def redshift_from_lookback_time(self, lookback_time, **parameters):
        # cut out times that are too large
        hubble_time = self.cosmology(parameters).hubble_time
        lookback_time *= (lookback_time <= hubble_time)
        
        # need to numerically invert the lookback time to get the redshift
        redshift_grid = xp.linspace(0, 100, 10000)
        lookback_time = lookback_time[:, : ,xp.newaxis]
        redshift_grid = redshift_grid[xp.newaxis, xp.newaxis, :]
        lookback_time_broadcast = lookback_time * xp.ones(redshift_grid.shape)
        redshift_grid_broadcast = redshift_grid * xp.ones(lookback_time.shape)
        
        # select the last time that is less than the target lookback time
        # then index redshift from that time
        times = self.cosmology(parameters).lookback_time(redshift_grid)
        mask = (times <= lookback_time) & (times != 0)
        closest_index = xp.count_nonzero(mask, axis=2, keepdims=True) - 1
        closest_redshift = xp.take_along_axis(redshift_grid_broadcast, closest_index, axis=2)
        return closest_redshift

    def psi_of_z(self, redshift, **parameters):
        r"""
        Madau-Dickinson model convolved with a time delay distribution.

        .. math::

            \psi(z|\kappa_d, \tau_{\mathrm{min}}) \propto \int d \tau SFR(Z(t_{\mathrm{merge}}(z) + \tau)) p(\tau)
            SFR(z) = \frac{(1 + z)^{2.7}{1 + (\frac{1 + z}{2.9})^{5.6}}
            p(\tau) = \tau^{-\kappa_d} (0 \leq \tau \leq \tau_{\mathrm{min}})

        Parameters
        ----------
        kappa_d: float
            Slope of the power-law time delay distribution, :math:`\kappa_d`.
        tau_min: float
            Lower cutoff of the time delay distribution, :math:`\tau_{\mathrm{min}}`. (Gyr)
        """
        kappa_d = parameters["kappa_d"]
        tau_min = parameters["tau_min"]
        
        hubble_time = self.cosmology(parameters).hubble_time
        tau_grid = xp.linspace(tau_min, hubble_time, 2500)
        
        SFR = MadauDickinsonRedshift().psi_of_z(self.redshift_from_lookback_time(
                        self.cosmology(parameters).lookback_time(redshift)[:, xp.newaxis] + tau_grid[xp.newaxis, :], **parameters), 
                                              gamma=2.7, z_peak=1.9, kappa=5.6)
        tau_probability = self.time_delay(tau_grid[xp.newaxis, :], **parameters)
                       
        psi_of_z = xp.trapz(SFR[:,:,0] * tau_probability, tau_grid, axis=1)
        
        # normalize it so psi(0) = 1
        SFR_z0 = MadauDickinsonRedshift().psi_of_z(self.redshift_from_lookback_time(
                        tau_grid[xp.newaxis, :], **parameters), 
                                                   gamma=2.7, z_peak=1.9, kappa=5.6)
        normalization = xp.trapz(SFR_z0[:,:,0] * tau_probability, tau_grid, axis=1)
        
        psi_of_z *= 1 / normalization
        
        return psi_of_z
