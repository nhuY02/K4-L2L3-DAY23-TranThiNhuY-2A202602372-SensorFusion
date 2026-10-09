"""Extended Kalman filter helpers for 6D constant-velocity motion.

Part E supplies prediction and correction for docs/HUONG_DAN_KY_THUAT.md §2.
Read the shared time step and process-noise settings with get_tracking_params().
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np
from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: Gợi ý module — params = get_tracking_params() sau khi import ở trên.


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build the constant-velocity state transition matrix F.

    Args:
        dt: Time step in seconds; default from tracking params.

    Returns:
        6x6 state transition matrix as ``np.matrix``.
    """
    # vi: TODO Part E — Nếu dt is None, lấy params.dt từ get_tracking_params().
    # vi: F = I_6; gán F[0,3]=F[1,4]=F[2,5]=dt (vị trí += v * dt).
    dt = get_tracking_params().dt if dt is None else float(dt)
    F = np.eye(6, dtype=float)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return np.asmatrix(F)


def build_Q(dt: Optional[float] = None, q: Optional[float] = None) -> Matrix:
    """Build the process noise covariance matrix Q.

    Args:
        dt: Time step; default from tracking params.
        q: Process noise scale; default from tracking params.

    Returns:
        6x6 process noise matrix.
    """
    # vi: TODO Part E — Q đường chéo: q_diag = dt * q trên 6 trục (mô hình lab).
    params = get_tracking_params()
    dt = params.dt if dt is None else float(dt)
    q = params.q if q is None else float(q)
    return np.asmatrix(np.eye(6, dtype=float) * (dt * q))


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance one time step forward.

    Args:
        x: State vector (6x1).
        P: State covariance (6x6).
        F: Optional transition matrix; build via ``build_F`` if None.
        Q: Optional process noise; build via ``build_Q`` if None.

    Returns:
        Tuple ``(x_pred, P_pred)``.
    """
    # vi: TODO Part E — x_pred = F @ x; P_pred = F @ P @ F.T + Q (dùng ma trận np).
    F = build_F() if F is None else np.asmatrix(F)
    Q = build_Q() if Q is None else np.asmatrix(Q)
    x_pred = F @ np.asmatrix(x)
    P_pred = F @ np.asmatrix(P) @ F.T + Q
    return np.asmatrix(x_pred), np.asmatrix(P_pred)


def innovation(x: Matrix, meas: Any) -> Matrix:
    """Compute the measurement residual (innovation) gamma.

    Args:
        x: Predicted state.
        meas: Measurement with ``z`` and ``sensor.get_hx(x)``.

    Returns:
        Innovation vector ``z - h(x)``.
    """
    # vi: TODO Part E — return meas.z - meas.sensor.get_hx(x).
    return np.asmatrix(meas.z) - np.asmatrix(meas.sensor.get_hx(x))


def innovation_covariance(P: Matrix, meas: Any, H: Matrix) -> Matrix:
    """Compute the innovation covariance S = H P H' + R.

    Args:
        P: State covariance.
        meas: Measurement with ``R``.
        H: Measurement Jacobian.

    Returns:
        Innovation covariance matrix S.
    """
    # vi: TODO Part E — S = H @ P @ H.T + meas.R (dùng @ với cả ndarray/matrix).
    H, P = np.asmatrix(H), np.asmatrix(P)
    return H @ P @ H.T + np.asmatrix(meas.R)


def ekf_update(x: Matrix, P: Matrix, meas: Any) -> tuple[Matrix, Matrix]:
    """Apply an EKF measurement update and return updated state and covariance.

    Args:
        x: Prior state.
        P: Prior covariance.
        meas: Associated measurement.

    Returns:
        Tuple ``(x_upd, P_upd)``.
    """
    # vi: TODO Part E — H = meas.sensor.get_H(x); gamma, S; K = P H' S^{-1};
    # vi: x_upd = x + K gamma; P_upd = (I - K H) P.
    H = np.asmatrix(meas.sensor.get_H(x))
    gamma = innovation(x, meas)
    S = innovation_covariance(P, meas, H)
    # Solve S K.T = (P H.T).T to avoid explicitly inverting S.
    PHt = np.asmatrix(P) @ H.T
    K = np.asmatrix(np.linalg.solve(np.asarray(S), np.asarray(PHt.T)).T)
    x_upd = np.asmatrix(x) + K @ gamma
    P_upd = (np.eye(np.asmatrix(P).shape[0]) - K @ H) @ np.asmatrix(P)
    return np.asmatrix(x_upd), np.asmatrix(P_upd)
