# =======================
# 실시간 PCA 안정화 클래스 written by dongmin. 2025.08.28
# =======================
class RealtimePCASmoother:
    
    def __init__(self, stability_thresh=1.5, power_iter_trigger=0.8):
        self.prev_axes = None
        self.frame_count = 0
        self.stability_thresh = stability_thresh  # 고유값 비율 임계치
        self.power_iter_trigger = power_iter_trigger  # Power iteration 트리거 임계치
        
    def power_iteration(self, cov_matrix, max_iter=3):

        """
        3D Power iteration for 경계박스 상당한 실시간 최적화.
        cov_matrix: (3,3) 공분산 행렬 (축 3개니까)
        max_iter: 최대 반복 횟수
        반환:
          main_axis(3,), main_eigenval 기본 주축과 고유값
        """
        n = cov_matrix.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(max_iter):
            v = cov_matrix @ v
            v = v / np.linalg.norm(v)
        
        # 고유값 추정
        eigenval = v.T @ cov_matrix @ v
        return v, eigenval
    
    def check_projection_stability(self, vals):
        """
        정사영 분산 체크 - 고유값 비율로 안정성 판단.
        vals: (3,) float32 : 고유값 : 내림차순
        return bool (true/false) (true : 안정적, false: 불안정)
        """

        if len(vals) < 3:
            return False
        ratio1 = vals[0] / vals[1] if vals[1] > 1e-6 else float('inf')
        ratio2 = vals[1] / vals[2] if vals[2] > 1e-6 else float('inf')
        return ratio1 > self.stability_thresh and ratio2 > self.stability_thresh
    
    def fast_axis_alignment(self, axes_new):
        """
        제한없는 임의 축과 기존 축의 내적 기반 계산. 센서 노이즈가 난무해도 안정적.
        axes_new: (3,3) 새 장축 행렬 (정규화된 벡터로 주축 감지됨)
        return (3,3) 정렬된 축 행렬
        """
        if self.prev_axes is None:
            self.prev_axes = axes_new.copy()
            return axes_new
        
        axes_aligned = axes_new.copy()
        # 벡터화된 내적 계산
        dots = np.sum(axes_new * self.prev_axes, axis=0)
        flip_mask = dots < 0
        axes_aligned[:, flip_mask] *= -1
        
        self.prev_axes = axes_aligned
        return axes_aligned
