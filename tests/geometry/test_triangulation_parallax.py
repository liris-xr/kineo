import math
import torch

from kineo.geometry.triangulation import triangulation_parallax_angles


def test_parallax_of_two_orthogonal_rays_is_a_right_angle():
    # Cameras on the x and z axes, point at the origin: the rays meet at 90 deg.
    camera_centers = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    points_3d = torch.zeros((1, 1, 3))

    angles = triangulation_parallax_angles(points_3d, camera_centers)

    assert angles.shape == (1, 1)
    assert math.isclose(angles[0, 0].item(), math.pi / 2, abs_tol=1e-6)


def test_views_that_did_not_observe_the_point_do_not_widen_its_parallax():
    # Two nearly coincident cameras plus a third that would subtend 90 deg but
    # scored nothing, so it contributed no ray to the triangulation.
    camera_centers = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.001, 0.0], [0.0, 0.0, 1.0]]]
    )
    points_3d = torch.zeros((1, 1, 3))
    points_weights = torch.tensor([[[0.9], [0.8], [0.0]]])

    angles = triangulation_parallax_angles(
        points_3d, camera_centers, points_weights
    )

    assert angles[0, 0].item() < math.radians(1.0)
