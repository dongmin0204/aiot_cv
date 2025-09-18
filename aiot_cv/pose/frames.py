import numpy as np

Array = np.ndarray

def to_hom(X: Array) -> Array:
    X = np.asarray(X)
    if X.ndim == 1:
        return np.concatenate([X, np.array([1.0])])
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([X, ones])

def from_hom(Xh: Array) -> Array:
    Xh = np.asarray(Xh)
    if Xh.ndim == 1:
        return Xh[:-1] / Xh[-1]
    w = Xh[:, -1:]
    return Xh[:, :-1] / w

def make_T(R: Array, t: Array) -> Array:
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert(T: Array) -> Array:
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.T
    ti = -Rt @ t
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = Rt
    Ti[:3, 3] = ti
    return Ti

def compose(T1: Array, T2: Array) -> Array:
    return T1 @ T2

def project_points(P_model: Array, T_co: Array, K: Array) -> Array:
    P_model = np.asarray(P_model, dtype=np.float64)
    R = T_co[:3, :3]
    t = T_co[:3, 3]
    Pc = (R @ P_model.T) + t[:, None]
    x, y, z = Pc[0], Pc[1], Pc[2]
    z = np.where(z == 0, 1e-9, z)
    u = (K[0,0] * (x / z)) + K[0,2]
    v = (K[1,1] * (y / z)) + K[1,2]
    return np.stack([u, v], axis=1)
