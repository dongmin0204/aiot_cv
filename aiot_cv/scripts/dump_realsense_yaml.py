import yaml, numpy as np

try:
	import pyrealsense2 as rs
except Exception:
	rs = None

def dump_realsense_yaml(outfile="aiot_cv/configs/camera.yaml"):
	assert rs is not None, "pyrealsense2 not installed"
	pipe = rs.pipeline(); cfg = rs.config()
	cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
	cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
	prof = pipe.start(cfg)
	color_p = prof.get_stream(rs.stream.color).as_video_stream_profile()
	depth_p = prof.get_stream(rs.stream.depth).as_video_stream_profile()
	Kc = color_p.get_intrinsics(); Kd = depth_p.get_intrinsics()
	ex_dc = depth_p.get_extrinsics_to(color_p)
	depth_sensor = prof.get_device().first_depth_sensor()
	depth_units = float(depth_sensor.get_depth_scale())
	rot = np.array(ex_dc.rotation).reshape(3,3)
	T = np.eye(4); T[:3,:3] = rot; T[:3,3] = np.array(ex_dc.translation)
	out = dict(
		version=1,
		device_serial=prof.get_device().get_info(rs.camera_info.serial_number),
		profile=dict(color=dict(width=Kc.width,height=Kc.height,fps=30),
					depth=dict(width=Kd.width,height=Kd.height,fps=30)),
		align_to_color=True,
		intrinsics=dict(
			color=dict(fx=Kc.fx, fy=Kc.fy, cx=Kc.ppx, cy=Kc.ppy, dist_model=str(Kc.model), dist=list(Kc.coeffs)),
			depth=dict(fx=Kd.fx, fy=Kd.fy, cx=Kd.ppx, cy=Kd.ppy, dist_model=str(Kd.model), dist=list(Kd.coeffs))
		),
		extrinsics=dict(T_color_from_depth=[list(T[0]), list(T[1]), list(T[2]), list(T[3])]),
		depth_units=depth_units,
	)
	pipe.stop()
	yaml.safe_dump(out, open(outfile, 'w'), sort_keys=False)
	print(f"saved: {outfile}")

if __name__ == '__main__':
	dump_realsense_yaml()
