# =======================
# 실시간 최적화된 3D PCA 경계 박스 계산 함수 written by dongmin. 2025.08.28
# =======================
def pca_obb_3d(points_xyz, smoother=None):
    """
    실시간 최적화된 3D PCA OBB 계산
    반환:
      center(3,), axes(3,3) [열벡터 u1,u2,u3 = 장축, 단축, 세번째축], lengths(3,), corners(8,3)
    - 장축 = u1 = X축
    - 단축 = u2 = Y축
    """
    pts = points_xyz.astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    
    # 기본 고유분해
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    
    # 정규화
    for k in range(3):
        n = np.linalg.norm(axes[:, k])
        if n > 0:
            axes[:, k] /= n
    
    # 실시간 안정화 적용
    if smoother is not None:
        smoother.frame_count += 1
        
        # 1) 정사영 분산 체크
        is_stable = smoother.check_projection_stability(vals)
        
        # 2) 불안정할 때만 Power iteration 적용
        if not is_stable and smoother.frame_count % 5 == 0:  # 5프레임마다 체크
            try:
                # 주축만 Power iteration으로 재계산
                main_axis, main_eigenval = smoother.power_iteration(cov, max_iter=3)
                axes[:, 0] = main_axis
            except:
                pass  # 실패시 기본 결과 사용
        
        # 3) 빠른 축 정렬 (항상 적용)
        axes = smoother.fast_axis_alignment(axes)
    
    # OBB 계산
    proj = centered @ axes
    mins = proj.min(axis=0); maxs = proj.max(axis=0)
    c_local = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5
    center = mean + axes @ c_local
    
    # 8개 꼭짓점 생성
    corners = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                corner = center + s1*half[0]*axes[:,0] + s2*half[1]*axes[:,1] + s3*half[2]*axes[:,2]
                corners.append(corner)
    corners = np.stack(corners, axis=0)
    lengths = 2.0 * half
    
    return center, axes, lengths, corners