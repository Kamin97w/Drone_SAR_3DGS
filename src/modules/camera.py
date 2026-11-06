"""Pinhole camera model and pixel -> world ray conversion.

Conventions (OpenCV / COLMAP):
  - Camera frame: x right, y down, z forward (along the optical axis).
  - Pixel (u, v): u to the right, v downward, in pixels.
  - Pose is stored as camera-to-world: R_c2w (3x3) and C (camera center in
    world coordinates). A world point X maps to the camera frame by
    X_cam = R_c2w.T @ (X - C).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Undistorted pinhole intrinsics. Images are assumed already rectified."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> "PinholeCamera":
        """Build intrinsics from the horizontal field of view.

        For the JKU rectified thermal images the paper gives 50.82 deg and
        512x512 px. Treat this as an approximation until you find the real
        calibration; check it against any intrinsics file in the dataset.
        """
        f = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0)

    def pixels_to_rays_cam(self, uv: np.ndarray) -> np.ndarray:
        """Unit ray directions in the camera frame for pixels uv, shape (N, 2)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        d = np.stack([x, y, np.ones_like(x)], axis=1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, X_cam: np.ndarray) -> np.ndarray:
        """Project camera-frame points (N, 3) to pixels (N, 2)."""
        X_cam = np.atleast_2d(X_cam)
        u = self.fx * X_cam[:, 0] / X_cam[:, 2] + self.cx
        v = self.fy * X_cam[:, 1] / X_cam[:, 2] + self.cy
        return np.stack([u, v], axis=1)


def rays_world(
    cam: PinholeCamera, R_c2w: np.ndarray, C: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame rays for pixels uv.

    Returns (origins, directions), both (N, 3); directions are unit length.
    """
    d_cam = cam.pixels_to_rays_cam(uv)
    d_world = d_cam @ np.asarray(R_c2w).T
    origins = np.broadcast_to(np.asarray(C, dtype=float), d_world.shape).copy()
    return origins, d_world


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion (COLMAP order)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_from_colmap(qvec_wxyz: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a COLMAP world-to-camera pose to (R_c2w, C).

    COLMAP stores X_cam = R_w2c @ X + t, so R_c2w = R_w2c.T and C = -R_w2c.T @ t.
    """
    R_w2c = quat_wxyz_to_matrix(qvec_wxyz)
    R_c2w = R_w2c.T
    C = -R_c2w @ np.asarray(tvec, dtype=float)
    return R_c2w, C