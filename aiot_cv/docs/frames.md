# Frames & Notation

- Right-handed coordinates, object → camera transform.
- T_{c<-o} = [R|t], with det(R)=+1.
- Output schema: {"R": (3,3) np.float64, "t": (3,), "T": (4,4), "frame": "camera"}.

## Projection
Given model points P_model (Nx3), pose T_{c<-o}, and intrinsics K:
- P_cam = R @ P_model.T + t[:,None]
- p = K @ [x/z, y/z, 1]
