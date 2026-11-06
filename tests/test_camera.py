import numpy as np

from modules.camera import PinholeCamera, pose_from_colmap, rays_world

# A drone 35 m up, looking straight down (OpenCV convention):
# camera x = world +X, camera y = world -Y, camera z = world -Z.
R_DOWN = np.diag([1.0, -1.0, -1.0])
C_DRONE = np.array([0.0, 0.0, 35.0])


def test_center_pixel_points_straight_down():
    cam = PinholeCamera.from_fov(512, 512, 50.82)
    _, d = rays_world(cam, R_DOWN, C_DRONE, np.array([[cam.cx, cam.cy]]))
    assert np.allclose(d[0], [0.0, 0.0, -1.0], atol=1e-9)


def test_edge_pixel_hits_ground_at_expected_offset():
    cam = PinholeCamera.from_fov(512, 512, 50.82)
    o, d = rays_world(cam, R_DOWN, C_DRONE, np.array([[0.0, cam.cy]]))
    t = -o[0, 2] / d[0, 2]          # intersect with the plane z = 0
    hit = o[0] + t * d[0]
    expected_x = -35.0 * np.tan(np.radians(50.82 / 2))   # about -16.6 m
    assert np.isclose(hit[0], expected_x, atol=1e-6)
    assert np.isclose(hit[2], 0.0, atol=1e-9)


def test_project_inverts_ray():
    cam = PinholeCamera.from_fov(512, 512, 50.82)
    uv = np.array([[100.0, 400.0]])
    d = cam.pixels_to_rays_cam(uv)
    assert np.allclose(cam.project(d * 7.0), uv)


def test_colmap_identity_pose():
    R, C = pose_from_colmap(np.array([1.0, 0, 0, 0]), np.zeros(3))
    assert np.allclose(R, np.eye(3)) and np.allclose(C, 0)


def test_colmap_center_roundtrip():
    q = np.array([0.9239, 0.0, 0.3827, 0.0])   # about 45 deg about y
    t = np.array([1.0, 2.0, 3.0])
    R_c2w, C = pose_from_colmap(q, t)
    # The camera center must map to the origin of the camera frame.
    assert np.allclose(R_c2w.T @ (C - C), 0)
    # And the world origin must land at t in the camera frame.
    assert np.allclose(R_c2w.T @ (np.zeros(3) - C), t, atol=1e-3)