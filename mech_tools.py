import numpy as np


def add_plane_stress_energy(mesh, E, nu, displacement_name="displacement"):
    mesh = mesh.copy()

    u = mesh.point_data[displacement_name]

    if u.shape[1] == 2:
        u = np.column_stack([u, np.zeros(len(u))])

    mesh.point_data["u"] = u

    grad_mesh = mesh.compute_derivative(
        scalars="u",
        gradient=True,
        preference="point",
    )

    grad_u = grad_mesh.point_data["gradient"].reshape((-1, 3, 3))

    strain = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))

    eps_xx = strain[:, 0, 0]
    eps_yy = strain[:, 1, 1]
    eps_xy = strain[:, 0, 1]

    coef = E / (1.0 - nu**2)

    sig_xx = coef * (eps_xx + nu * eps_yy)
    sig_yy = coef * (nu * eps_xx + eps_yy)
    sig_xy = E / (1.0 + nu) * eps_xy

    energy_density = 0.5 * (
        sig_xx * eps_xx
        + sig_yy * eps_yy
        + 2.0 * sig_xy * eps_xy
    )

    mesh.point_data["strain_xx"] = eps_xx
    mesh.point_data["strain_yy"] = eps_yy
    mesh.point_data["strain_xy"] = eps_xy
    mesh.point_data["internal_energy_density"] = energy_density

    return mesh


def total_internal_energy(mesh):
    cell_mesh = mesh.point_data_to_cell_data()
    cell_mesh = cell_mesh.compute_cell_sizes(length=False, area=True, volume=False)

    A = cell_mesh.cell_data["Area"]
    w = cell_mesh.cell_data["internal_energy_density"]

    return np.sum(w * A)