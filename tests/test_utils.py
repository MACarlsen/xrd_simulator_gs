import unittest
import warnings

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from xfab import tools

from xrd_simulator import utils

rng = np.random.default_rng(0)


class TestUtils(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)  # changes all randomisation in the test

    def test_sample_convex_hull_3d(self):
        hull = ConvexHull(rng.random((100, 3)))
        points = hull.points
        r = points.mean(axis=0)
        tris = points[hull.simplices]
        a = tris[:, 0] - r
        b = tris[:, 1] - r
        c = tris[:, 2] - r
        vols = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
        cents = (r + tris[:, 0] + tris[:, 1] + tris[:, 2]) / 4.0
        cents = np.sum(cents * vols[:, None], axis=0) / vols.sum()
        X = utils._sample_convex_hull_3d(hull, 200000)
        A = hull.equations[:, :-1]
        b = hull.equations[:, -1]
        assert np.all(X @ A.T + b <= 1e-10)
        assert np.allclose(X.mean(axis=0), cents, atol=1e-2)

    def test_clip_line_with_convex_polyhedron(self):
        line_points = np.ascontiguousarray([[-1.0, 0.2, 0.2], [-1.0, 0.4, 0.6]])
        line_direction = np.ascontiguousarray([1.0, 0.0, 0.0])
        line_direction = line_direction / np.linalg.norm(line_direction)
        plane_points = np.ascontiguousarray(
            [
                [0.0, 0.5, 0.5],
                [1, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0, 0.5],
                [0.5, 1.0, 0.5],
            ]
        )
        plane_normals = np.ascontiguousarray(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        clip_lengths = utils._clip_line_with_convex_polyhedron(
            line_points, line_direction, plane_points, plane_normals
        )
        for clip_length in clip_lengths:
            self.assertAlmostEqual(
                clip_length,
                1.0,
                msg="Projection through unity cube should give unity clip length",
            )

        line_direction = np.ascontiguousarray([1.0, 0.2, 0.1])
        line_direction = line_direction / np.linalg.norm(line_direction)
        clip_lengths = utils._clip_line_with_convex_polyhedron(
            line_points, line_direction, plane_points, plane_normals
        )
        for clip_length in clip_lengths:
            self.assertGreater(
                clip_length,
                1.0,
                msg="Tilted projection through unity cube should give greater than unity clip length",
            )

    def test_lab_strain_to_B_matrix(self):
        U = Rotation.random().as_matrix()
        strain_tensor = (np.random.rand(3, 3) - 0.5) * 1e-2  # random strain tensor
        strain_tensor = (strain_tensor.T + strain_tensor) / 2.0
        unit_cell = [5.028, 5.028, 5.519, 90.0, 90.0, 120.0]

        B0 = tools.form_b_mat(unit_cell)
        B = utils._lab_strain_to_B_matrix(strain_tensor, U, B0).squeeze()

        n_c = np.random.rand(
            3,
        )  # crystal unit vector
        n_c = n_c / np.linalg.norm(n_c)
        n_l = np.dot(U, n_c)  # lab unit vector

        # strain along n_l described in lab frame
        strain_l = np.dot(np.dot(n_l, strain_tensor), n_l)
        s = utils.ensure_numpy(utils._b_to_epsilon(B, B0))
        crystal_strain = np.array(
            [[s[0], s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], s[5]]]
        )

        # strain along n_l described in crystal frame
        strain_c = np.dot(np.dot(n_c, crystal_strain), n_c)

        # The strain should be invariant along a direction
        self.assertAlmostEqual(
            strain_l, strain_c, msg="bad crystal to lab frame conversion"
        )

    def test_alpha_to_quarternion(self):
        _, alpha_2, alpha_3 = np.random.rand(
            3,
        )
        q = utils._alpha_to_quarternion(0, alpha_2, alpha_3)
        self.assertAlmostEqual(q[0], 1.0, msg="quarternion wrongly computed")
        self.assertAlmostEqual(q[1], 0.0, msg="quarternion wrongly computed")
        self.assertAlmostEqual(q[2], 0.0, msg="quarternion wrongly computed")
        self.assertAlmostEqual(q[3], 0.0, msg="quarternion wrongly computed")
        alpha_1 = np.random.rand(
            7,
        )
        alpha_2 = np.random.rand(
            7,
        )
        alpha_3 = np.random.rand(
            7,
        )
        qq = utils._alpha_to_quarternion(alpha_1, alpha_2, alpha_3)
        for q in qq:
            self.assertTrue(
                np.abs(np.linalg.norm(q) - 1.0) < 1e-5, msg="quarternion not normalised"
            )

    def test_epsilon_to_b(self):
        unit_cell = [4.926, 4.926, 5.4189, 90.0, 90.0, 120.0]
        eps1 = (
            25
            * 1e-4
            * (
                np.random.rand(
                    6,
                )
                - 0.5
            )
        )
        B0 = tools.form_b_mat(unit_cell)
        strain_tensor1 = utils._strain_as_tensor(eps1)
        B = utils.ensure_numpy(utils._epsilon_to_b(strain_tensor1, B0))
        eps2 = utils._b_to_epsilon(B.reshape(1, 3, 3), B0)
        eps2 = utils.ensure_numpy(eps2.squeeze())
        self.assertTrue(np.allclose(eps1, eps2))

    def test_get_misorientations(self):
        orientations = np.zeros((2, 3, 3))
        orientations[0, :, :] = np.eye(3)
        c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
        orientations[1, :, :] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        misorientations = utils._get_misorientations(orientations)
        self.assertEqual(misorientations.shape[0], 2)
        self.assertAlmostEqual(misorientations[0], np.radians(5.0))
        self.assertAlmostEqual(misorientations[1], np.radians(5.0))

        orientations = np.zeros((2, 3, 3))
        orientations[0, :, :] = np.eye(3)
        orientations[1, :, :] = np.eye(3)
        misorientations = utils._get_misorientations(orientations)
        self.assertEqual(misorientations.shape[0], 2)
        self.assertAlmostEqual(misorientations[0], 0)
        self.assertAlmostEqual(misorientations[1], 0)

    def test_diffractogram_deprecated(self):
        """Test that _diffractogram raises a deprecation warning.

        .. deprecated::
            This test verifies that _diffractogram is properly marked as deprecated.
            The function will be removed in a future version.
        """
        diffraction_pattern = np.zeros((20, 20))
        R = 8
        det_c_z, det_c_y = 10.0, 10.0
        for i in range(diffraction_pattern.shape[0]):
            for j in range(diffraction_pattern.shape[1]):
                if np.abs(np.sqrt((i - det_c_z) ** 2 + (j - det_c_y) ** 2) - R) < 0.5:
                    diffraction_pattern[i, j] += 1

        # Verify deprecation warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_centres, histogram = utils._diffractogram(
                diffraction_pattern, det_c_z, det_c_y, 1.0
            )
            # Check that deprecation warning was raised
            self.assertTrue(
                any(issubclass(warning.category, DeprecationWarning) for warning in w),
                msg="_diffractogram should raise DeprecationWarning",
            )

        # Verify function still works correctly (for backward compatibility)
        self.assertEqual(
            np.sum(histogram > 0), 1, msg="Error in diffractogram azimuth integration"
        )
        self.assertEqual(
            np.sum(histogram),
            np.sum(diffraction_pattern),
            msg="Error in diffractogram azimuth integration",
        )

    def test_contained_by_intervals_deprecated(self):
        """Test that _contained_by_intervals raises a deprecation warning.

        .. deprecated::
            This test verifies that _contained_by_intervals is properly marked as deprecated.
            The function will be removed in a future version.
        """
        intervals = [[0.0, 0.5], [0.7, 1.0]]

        # Verify deprecation warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = utils._contained_by_intervals(0.3, intervals)
            # Check that deprecation warning was raised
            self.assertTrue(
                any(issubclass(warning.category, DeprecationWarning) for warning in w),
                msg="_contained_by_intervals should raise DeprecationWarning",
            )

        # Verify function still works correctly (for backward compatibility)
        self.assertTrue(result, msg="0.3 should be contained in [0.0, 0.5]")

    def test_reciprocal_metric_for_strain_conversion(self):
        # Test reciprocal metric is preserved in strain mapping to
        # recirprocal crystal basis matrix B. The idea is that this should
        # work for Green-Lagrange strain defined as E = 0.5*(F.T@F -I)
        # where I is idenity matrix and F the deformation gradient tensor.
        # since it is possible to show that F = B^-T @ B0^T
        # we have that C = F.T@F =  B0 @ B^-1 @ B^-T @ B0^T, and thus we have
        # that B^-1 @ B^-T = B0^-1 @ (2*E + I) @ B0^-T and thus
        # that B^T @ B = B0^T @ (2*E + I)^-1 @ B0 or equivalently
        # that B^T @ B = B0^T @ C^-1 @ B0

        unit_cell = [1.12365, 2.34897, 3.23874, 90.234, 110.12, 120.35]
        B0 = tools.form_b_mat(unit_cell)

        E = np.array(
            [
                [0.001, 0.004, -0.002],
                [0.004, -0.0005, 0.003],
                [-0.002, 0.003, 0.002],
            ],
        )

        B = utils.ensure_numpy(utils._epsilon_to_b(E.reshape(1, 3, 3), B0).squeeze(0))

        C = np.eye(3) + 2.0 * E

        expected_reciprocal_metric = B0.T @ np.linalg.solve(C, B0)

        actual_reciprocal_metric = B.T @ B

        np.testing.assert_allclose(
            actual_reciprocal_metric,
            expected_reciprocal_metric,
            atol=1e-12,
            rtol=1e-12,
        )

        # let us also test a batch of strain tensors
        E = np.concatenate(
            [
                E[None, :, :],
                1.752 * E[None, :, :],
                -2.234 * E[None, :, :],
                -0.134 * E[None, :, :],
            ],
            axis=0,
        )
        E[-1, -1, -1] -= 0.00084237
        E[-1, 0, 1] += 0.00124
        E[-1, 1, 0] += 0.00124

        B = utils._epsilon_to_b(E, B0).numpy()

        self.assertEqual(B.shape[0], 4)
        self.assertEqual(B.shape[1], 3)
        self.assertEqual(B.shape[2], 3)

        for i in range(B.shape[0]):
            C = np.eye(3) + 2.0 * E[i]

            expected_reciprocal_metric = B0.T @ np.linalg.solve(C, B0)

            actual_reciprocal_metric = B[i].T @ B[i]

            np.testing.assert_allclose(
                actual_reciprocal_metric,
                expected_reciprocal_metric,
                atol=1e-12,
                rtol=1e-12,
            )

    def test_zero_strain_returns_B0(self):
        unit_cell = [1.12365, 2.34897, 3.23874, 90.234, 110.12, 120.35]
        B0 = tools.form_b_mat(unit_cell)

        B = utils.ensure_numpy(utils._epsilon_to_b(np.zeros((3, 3)), B0)).squeeze()

        np.testing.assert_allclose(B, B0, atol=1e-12, rtol=1e-12)

    def test_strain_B_batch_roundtrip(self):
        unit_cell = [1.12365, 2.34897, 3.23874, 90.234, 110.12, 120.35]
        B0 = tools.form_b_mat(unit_cell)

        E = np.array(
            [
                [
                    [0.01, 0.03, -0.02],
                    [0.03, -0.01, 0.015],
                    [-0.02, 0.015, 0.02],
                ],
                [
                    [-0.02, -0.01, 0.025],
                    [-0.01, 0.03, -0.02],
                    [0.025, -0.02, 0.01],
                ],
            ]
        )

        B = utils._epsilon_to_b(E, B0)
        recovered = utils._b_to_epsilon(B, B0)

        for i in range(len(E)):
            np.testing.assert_allclose(
                utils._strain_as_tensor(recovered[i]),
                E[i],
                atol=1e-12,
                rtol=1e-12,
            )

    def test_lab_strain_preserves_principal_reflection_direction(self):
        B0 = np.array(
            [
                [5.0, -0.8, 0.4],
                [0.0, 4.2, 0.7],
                [0.0, 0.0, 3.6],
            ],
            dtype=float,
        )

        U = Rotation.from_rotvec([0.4, -0.2, 0.3]).as_matrix()
        hkl = np.array([1.0, 1.0, 1.0])

        G0_lab = U @ B0 @ hkl
        n_lab = G0_lab / np.linalg.norm(G0_lab)

        axial_strain = 0.1
        transverse_strain = -0.05

        E_lab = transverse_strain * np.eye(3) + (
            axial_strain - transverse_strain
        ) * np.outer(n_lab, n_lab)

        B = utils.ensure_numpy(utils._lab_strain_to_B_matrix(E_lab, U, B0)).squeeze()

        G_lab = U @ B @ hkl

        np.testing.assert_allclose(
            G_lab / np.linalg.norm(G_lab),
            G0_lab / np.linalg.norm(G0_lab),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_B_preserves_principal_reflection_direction_general_lattice(self):
        B0 = np.array(
            [
                [5.0, -0.8, 0.4],
                [0.0, 4.2, 0.7],
                [0.0, 0.0, 3.6],
            ],
            dtype=float,
        )

        hkl = np.array([1.0, 1.0, 1.0])

        G0 = B0 @ hkl
        n = G0 / np.linalg.norm(G0)

        # Axisymmetric Green-Lagrange strain with n as a principal direction.
        axial_strain = 0.1
        transverse_strain = -0.5 * axial_strain

        E = axial_strain * np.outer(n, n) + transverse_strain * (
            np.eye(3) - np.outer(n, n)
        )

        B = utils.ensure_numpy(utils._epsilon_to_b(E, B0)).squeeze()

        G = B @ hkl

        assert np.allclose(
            G / np.linalg.norm(G),
            G0 / np.linalg.norm(G0),
            atol=1e-12,
        )

    def test_epsilon_to_b_rejects_bad_strain(self):
        B0 = np.eye(3)
        E = np.diag([0.0, 0.0, -0.5])  # C = I + 2E is singular

        with self.assertRaises(ValueError):
            utils._epsilon_to_b(E, B0)


if __name__ == "__main__":
    unittest.main()
