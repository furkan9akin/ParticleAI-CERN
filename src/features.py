import numpy as np
import awkward as ak


def extract_photon_features(events):
    """
    Extract kinematic features from the two
    highest-pT photons in each event.

    Momentum units are assumed to be GeV.
    """

    # Sort photon candidates by transverse momentum
    order = ak.argsort(
        events["photon_pt"],
        axis=1,
        ascending=False
    )

    pt = ak.values_astype(
        events["photon_pt"][order], np.float64
    )
    eta = ak.values_astype(
        events["photon_eta"][order], np.float64
    )
    phi = ak.values_astype(
        events["photon_phi"][order], np.float64
    )

    # Keep events with at least two photons
    valid = ak.num(pt) >= 2

    pt = pt[valid]
    eta = eta[valid]
    phi = phi[valid]

    # Leading and subleading photons
    pt1 = pt[:, 0]
    pt2 = pt[:, 1]

    eta1 = eta[:, 0]
    eta2 = eta[:, 1]

    phi1 = phi[:, 0]
    phi2 = phi[:, 1]

    # Angular differences
    delta_eta = eta1 - eta2
    delta_phi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi

    # Diphoton invariant mass
    mass_squared = (
        2 * pt1 * pt2 *
        (np.cosh(delta_eta) - np.cos(delta_phi))
    )

    diphoton_mass = np.sqrt(
        np.maximum(mass_squared, 0)
    )

    # Diphoton transverse momentum
    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)

    diphoton_pt = np.sqrt(px**2 + py**2)

    return {
        "lead_photon_pt_GeV": ak.to_numpy(pt1),
        "sublead_photon_pt_GeV": ak.to_numpy(pt2),
        "lead_photon_eta": ak.to_numpy(eta1),
        "sublead_photon_eta": ak.to_numpy(eta2),
        "delta_eta_gg": ak.to_numpy(delta_eta),
        "delta_phi_gg": ak.to_numpy(delta_phi),
        "diphoton_pt_GeV": ak.to_numpy(diphoton_pt),
        "diphoton_mass_GeV": ak.to_numpy(diphoton_mass),
    }
