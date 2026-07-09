from __future__ import annotations

import torch
import numpy as np


def _lorentz(
    k_in: torch.Tensor, k_out: torch.Tensor, rot_axis: torch.Tensor
) -> torch.Tensor | float:
    """Compute the Lorentz intensity factor for all reflections.

    This function calculates the Lorentz factor for X-ray diffraction based on
    the incident beam direction, scattered wave vectors, and rotation axis.

    Parameters
    ----------
    k_in : torch.Tensor
        Incident beam direction vector with shape (3,)
    k_out : torch.Tensor
        Scattered wave vectors with shape (N, 3) or (3,)
    rot_axis : torch.Tensor
        Rotation axis vector with shape (3,)

    Returns
    -------
    torch.Tensor | float
        Lorentz factors for each reflection with shape (N,) or a single float.
        Returns infinity for geometrically impossible reflections.

    Examples
    --------
    >>> k_in = torch.tensor([1., 0., 0.])
    >>> k_out = torch.tensor([[0.5, 0.5, 0.], [0.2, 0.3, 0.1]])
    >>> rot_axis = torch.tensor([0., 0., 1.])
    >>> lorentz(k_in, k_out, rot_axis)
    tensor([1.1547, 1.0758])

    Notes
    -----
    The Lorentz factor is calculated as :math:`1/|\\mathbf{\\omega}\\cdot\\mathbf{k}_{\\mathrm{in}}\\times\\mathbf{q}|`,
    where :math:`\\mathbf{\\omega}` is the the rotation axis and :math:`\\mathbf{q}` is the scattering plane normal.
    """

    # re-shape if single vector is given
    kp = k_out.reshape(-1, 3) if k_out.dim() == 1 else k_out

    k_norm = torch.linalg.norm(k_in)
    q = kp - k_in[None :]
    one_over_l = torch.matmul(torch.cross(k_in, rot_axis), q.T) * (1/k_norm**2)

    # Avoiding div by zero errors
    tol=torch.Tensor([0.5]) # Degrees
    condition = one_over_l > 1 / torch.sin(torch.deg2rad(tol))
    result = 1/one_over_l
    result = torch.where(condition, torch.tensor(float("inf")), result)
    return result.squeeze()


def _polarization(k_out: torch.Tensor, pol_vec: torch.Tensor) -> torch.Tensor | float:
    """Compute the Polarization intensity factor for all reflections.

    This function calculates the polarization factor for X-ray diffraction based on
    the polarization vector of the incident beam and the scattered wave vectors.

    Parameters
    ----------
    k_out : torch.Tensor
        Scattered wave vectors with shape (N, 3) or (3,)
    pol_vec : torch.Tensor
        Polarization vector with shape (3,)


    Returns
    -------
    torch.Tensor | float
        Polarization factors for each reflection with shape (N,) or a single float

    Examples
    --------
    >>> k_out = torch.tensor([[0.5, 0.5, 0.], [0., 0.3, 0.7]])
    >>> pol_vec = torch.tensor([0., 0., 1.])
    >>> polarization(pol_vec, k_out)
    tensor([1.0000, 0.1552])

    Notes
    -----
    The polarization factor is calculated as :math:`1 - (\\vec{p} \\cdot \\hat{k})^2`,
    where :math:`\\vec{p}` is the polarization vector and :math:`\\hat{k}` is the
    normalized scattered wave vector.
    """
    kp = k_out.reshape(-1, 3) if k_out.dim() == 1 else k_out
    # Normalize each k_out vector
    kp_norm = kp / torch.linalg.norm(kp, dim=1).reshape(-1, 1)
    # Calculate dot product between pol_vec and each normalized k_out
    dot_products = torch.matmul(kp_norm, pol_vec)
    return 1 - dot_products**2


def _scherrer(
    volumes: torch.Tensor, two_theta: torch.Tensor, wavelength: float, K: float = 0.9
) -> torch.Tensor:
    """Calculate Scherrer peak broadening FWHM.

    Parameters
    ----------
    volumes : torch.Tensor
        Crystallite volumes in cubic microns.
    two_theta : torch.Tensor
        Scattering angles in radians.
    wavelength : float
        X-ray wavelength in Angstroms.
    K : float, optional
        Scherrer shape factor. Default is 0.9 for spherical crystallites.

    Returns
    -------
    torch.Tensor
        Peak FWHM in radians.
    """

    crystallite_size = (
        2.0 * (3 * volumes / (4 * np.pi)) ** (1 / 3) * 10_000
    )  # Convert microm to Angstroms
    return K * wavelength / (crystallite_size * torch.cos(two_theta / 2))
