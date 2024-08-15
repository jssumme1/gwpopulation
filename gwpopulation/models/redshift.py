"""
Implemented redshift models
"""

import numpy as xp
import scipy.special as scs

from ..experimental.cosmo_models import CosmoMixin
from ..utils import powerlaw, inverse_cdf_time_delay

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
    
    def set_max_redshift(self, z):
        self.max_redshift = z
        
    def set_fast(self, boolean):
        self.fast = boolean
        
    def set_num_samples(self, n):
        self.num_samples = int(n)
    
    def time_delay(self, tau, **parameters):
        kappa_d = parameters["kappa_d"]
        tau_min = parameters["tau_min"]
        return powerlaw(tau, kappa_d, self.hubble_time, tau_min)
    
    def MadauDickinson_SFR(self, redshift):
        gamma = 2.7
        kappa = 5.6
        z_peak = 1.9
        
        psi_of_z = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** kappa
        )
        psi_of_z *= 1 + (1 + z_peak) ** (-kappa)
        return psi_of_z
                        
    def redshift_from_lookback_time(self, lookback_time):
        z = xp.interp(lookback_time, self.lookback_time_grid, self.redshift_grid, left=0, right=-1)
        return z
    
    def lookback_time_from_redshift(self, redshift):
        lookback_time = xp.interp(redshift, self.redshift_grid, self.lookback_time_grid)
        return lookback_time
    
    def integrand(self, x, redshift, **parameters):
        return x
        
    def psi_of_z(self, redshift, num_samples=5000, max_redshift=1000, fast=True, **parameters):
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
        if "jax" in xp.__name__:
            from jax import random
                        
        # allow these to be changed for popstock
        if hasattr(self, 'max_redshift'):
            max_redshift = self.max_redshift
        if hasattr(self, 'fast'):
            fast = self.fast
        if hasattr(self, 'num_samples'):
            num_samples = self.num_samples
                    
        kappa_d = parameters["kappa_d"]
        tau_min = parameters["tau_min"]
        if not hasattr(self, 'hubble_time'):
            #self.hubble_time = self.cosmology(parameters).hubble_time.value
            self.hubble_time = self.cosmology(parameters).lookback_time(100)
            
        # save redshift grid and lookback time grid
        if not hasattr(self, 'redshift_grid'):
            self.redshift_grid = xp.logspace(-3, 2, 10000)
        if not hasattr(self, 'lookback_time_grid'):
            self.lookback_time_grid = self.cosmology(parameters).lookback_time(self.redshift_grid)
                
        # draw the samples once and save them
        if hasattr(self, 'uniform_samples'):
            if len(self.uniform_samples) != num_samples:
                draw_samples = True
            else:
                draw_samples = False
        else:
            draw_samples = True
            
        if draw_samples == True:
            if "jax" not in xp.__name__:
                rng = xp.random.default_rng()
                self.uniform_samples = rng.random(int(num_samples))
            else:
                key = random.PRNGKey(758493)
                self.uniform_samples = random.uniform(key, shape=(int(num_samples),))
                        
        merger_lookback_time = self.lookback_time_from_redshift(redshift)
        
        if fast == False:
            tau_samples = inverse_cdf_time_delay(self.uniform_samples[xp.newaxis, ...], 
                                                 kappa_d, self.hubble_time - merger_lookback_time[..., xp.newaxis], tau_min)
            t_form = merger_lookback_time[..., xp.newaxis] + tau_samples    
                            
        else:
            tau_samples = inverse_cdf_time_delay(self.uniform_samples, kappa_d, self.hubble_time, tau_min)
            t_form = merger_lookback_time[..., xp.newaxis] + tau_samples[xp.newaxis, ...]

        # calculate z_form incrementally. jax dies if a large array is interpolated all at once.
        ns = 30 # number of slices
        ss = int(t_form.shape[0]/ns) # size of slice
        for ii in range(ns):
            if ii == 0: # first slice
                z_form = self.redshift_from_lookback_time(t_form[0:ss,...])
            elif ii == ns-1: # last slice
                z_form = xp.concatenate((z_form, self.redshift_from_lookback_time(t_form[(ii*ss):,...])), axis=0)
            else: # intermediate slices
                z_form = xp.concatenate((z_form, self.redshift_from_lookback_time(t_form[(ii*ss):(ii*ss+ss),...])), axis=0)

            
        # calculate SFR, eliminating values with lookback time > hubble time 
        SFR = xp.where(z_form != -1, self.MadauDickinson_SFR(z_form), xp.zeros(z_form.shape))
        SFR = xp.where(z_form <= max_redshift, SFR, 0)
        # adds in metallicity dependence, if relevant
        weighted_SFR = self.integrand(SFR, z_form, **parameters)    
        
        psi_of_z = xp.sum(weighted_SFR, axis=-1) / num_samples
        
        # normalize it
        psi_of_z /= self.normalize_psi(num_samples = num_samples, **parameters)        
        
        return psi_of_z

    def normalize_psi(self, num_samples=5000, max_redshift=1000, **parameters):
        r'''
        The normalized version of psi_of_z can be obtained by
        psi_of_z / normalize_psi
        '''
        kappa_d = parameters["kappa_d"]
        tau_min = parameters["tau_min"]
        tau_samples = inverse_cdf_time_delay(self.uniform_samples, kappa_d, self.hubble_time, tau_min)
        t_form = tau_samples
        z_form = self.redshift_from_lookback_time(t_form)
        SFR = self.MadauDickinson_SFR(z_form)  
        SFR = xp.where(z_form <= max_redshift, SFR, 0)
        # broadcast for safety in case metallicity weight is included
        weighted_SFR = self.integrand(SFR, z_form, **parameters)
        psi_of_z = xp.sum(weighted_SFR, axis=-1) / num_samples
                        
        return psi_of_z


class HeavisideRedshift(_Redshift):
    base_variable_names = ["zmin", "zmax"]

    def psi_of_z(self, redshift, **parameters):
        r"""
        Heaviside function redshift
        
        .. math::
        
            \psi(z|z_{\rm min}, z_{\rm max}) = H(z-z_{\rm min}) - H(z - z_{\rm max})

        Parameters
        ----------
        zmin: float
            minimum value of redhisft, :math:`z_{\rm min}`.
        zmax: float
            maximum value of redshift, :math:`z_{\rm max}`.
        """
        return xp.ones(redshift.shape) * (redshift >= parameters["zmin"]) * (redshift <= parameters["zmax"])

class MadauDickinsonHeavisideRedshift(MadauDickinsonRedshift):
    base_variable_names = ["gamma", "z_peak", "kappa", "zmin", "zmax"]
    
    def psi_of_z(self, redshift, **parameters):
        zmin = parameters["zmin"]
        zmax = parameters["zmax"]
        
        full_psi_of_z = super().psi_of_z(redshift, **parameters)
        return full_psi_of_z * (redshift >= zmin) * (redshift <= zmax)
    
class TimeDelayHeavisideRedshift(TimeDelayRedshift):
    def psi_of_z(self, redshift, num_samples=5000, **parameters):
        zmin = parameters["zmin"]
        zmax = parameters["zmax"]
        
        full_psi_of_z = super().psi_of_z(redshift, num_samples=num_samples, **parameters)
        return full_psi_of_z * (redshift >= zmin) * (redshift <= zmax)
    

class MetallicityWeightedTimeDelayRedshift(TimeDelayRedshift):
    base_variable_names = ["kappa_d", "tau_min", "Zmax"]
    def integrand(self, x, redshift, **parameters):
        Zmax = parameters["Zmax"] # Z / Z_sun
        frac_star_formation =  scs.gammainc(0.84, Zmax**2 * 10**(0.3 * redshift)) / 1.122
        return x * frac_star_formation
    

class MadauDickinsonSwitchRedshift(_Redshift):

    base_variable_names = ["gamma", "kappa", "z_peak", "eta", "z_turn", "boost"]

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
        z_turn: float
            Redshift at which power law slope changes to eta
        eta: float
            high-redshift power law slope
        boost: float
            multiplicative factor applied to high-z merger rate
        """
        gamma = parameters["gamma"]
        kappa = parameters["kappa"]
        z_peak = parameters["z_peak"]
        z_turn = parameters["z_turn"]
        eta = parameters["eta"]
        boost = parameters["boost"]
        
        MDR = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** kappa
        ) * (1 + (1 + z_peak) ** (-kappa))
        
        highz = (1 + redshift) ** eta
        
        match1 = (1 + z_turn) ** gamma / (
            1 + ((1 + z_turn) / (1 + z_peak)) ** kappa
        ) * (1 + (1 + z_peak) ** (-kappa))
        match2 = (1 + z_turn) ** eta
        
        highz *= match1 / match2 * boost
        
        psi_of_z = MDR * (redshift <= z_turn) + highz * (redshift > z_turn)
        
        return psi_of_z
