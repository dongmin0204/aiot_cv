# CV

[GitHub - dongmin0204/FoundationPose: [CVPR 2024 Highlight] FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://github.com/dongmin0204/FoundationPose)

- Foundation Pose 모델 레포 분석
    
    ```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                           FOUNDATIONPOSE PIPELINE                              │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   INPUT DATA    │    │   3D MODEL      │    │   CAMERA INFO   │
    │                 │    │                 │    │                 │
    │ • RGB Image     │    │ • Mesh (.obj)   │    │ • Intrinsic K   │
    │   (H, W, 3)     │    │ • Vertices      │    │   (3, 3)        │
    │ • Depth Image   │    │ • Normals       │    │ • Resolution    │
    │   (H, W)        │    │ • Textures      │    │   (H, W)        │
    │ • Object Mask   │    │                 │    │                 │
    │   (H, W) [초기] │    │                 │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │    PREPROCESSING        │
                        │                         │
                        │ • Depth Filtering       │
                        │   - Erosion (radius=2)  │
                        │   - Bilateral Filter    │
                        │ • XYZ Map Generation    │
                        │ • Mesh Processing       │
                        │   - Center to Origin    │
                        │   - Compute Diameter    │
                        │   - Voxel Downsample    │
                        └─────────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────────────────┐
                │                POSE ESTIMATION                     │
                │              (First Frame Only)                    │
                └────────────────────────────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │  INITIAL POSE GUESS     │
                        │                         │
                        │ 1. Rotation Grid        │
                        │    • Icosphere Views    │
                        │    • Inplane Rotations  │
                        │    • Symmetry Cluster   │
                        │      → (N_views, 4, 4)  │
                        │                         │
                        │ 2. Translation Guess    │
                        │    • Mask Center (u,v)  │
                        │    • Median Depth (z)   │
                        │    • K^-1 * [u,v,1] * z │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     POSE REFINER        │
                        │    (Iteration × 5)      │
                        │                         │
                        │ For each pose candidate:│
                        │                         │
                        │ Input Preparation:      │
                        │ • Crop around 3D bbox   │
                        │ • Resize to 224×224     │
                        │ • Render synthetic view │
                        │                         │
                        │ RefineNet Forward:      │
                        │ A: Render (B,6,224,224) │
                        │ B: Real   (B,6,224,224) │
                        │ ├─ RGB (3 channels)     │
                        │ └─ XYZ (3 channels)     │
                        │                         │
                        │ Output:                 │
                        │ • trans_delta: (B, 3)   │
                        │ • rot_delta: (B, 3)     │
                        │                         │
                        │ Update:                 │
                        │ pose_new = pose_old +   │
                        │           delta_pose    │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     SCORE NETWORK       │
                        │                         │
                        │ For all refined poses:  │
                        │                         │
                        │ Input Preparation:      │
                        │ • Same cropping process │
                        │ • Batch all candidates  │
                        │                         │
                        │ ScoreNet Forward:       │
                        │ A: Render (B*L,6,224,224)│
                        │ B: Real   (B*L,6,224,224)│
                        │ L: number of candidates │
                        │                         │
                        │ Multi-head Attention:   │
                        │ • Self-attention on     │
                        │   image features        │
                        │ • Cross-attention       │
                        │   between candidates    │
                        │                         │
                        │ Output:                 │
                        │ • scores: (B, L)        │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │    BEST POSE SELECTION  │
                        │                         │
                        │ • Sort by score         │
                        │ • Select highest        │
                        │ • Store for tracking    │
                        │                         │
                        │ best_pose = poses[      │
                        │   scores.argmax()]      │
                        └─────────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────────────────┐
                │                    TRACKING                        │
                │               (Subsequent Frames)                  │
                └────────────────────────────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   POSE REFINEMENT       │
                        │    (Iteration × 2)      │
                        │                         │
                        │ Input:                  │
                        │ • Previous frame pose   │
                        │ • Current RGB-D         │
                        │                         │
                        │ Process:                │
                        │ • Single pose input     │
                        │ • Same RefineNet        │
                        │ • Fewer iterations      │
                        │                         │
                        │ Output:                 │
                        │ • Refined pose          │
                        │ • Update pose_last      │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     FINAL OUTPUT        │
                        │                         │
                        │ • 6DOF Pose Matrix      │
                        │   (4, 4) transformation │
                        │                         │
                        │ • Visualization         │
                        │   - 3D bounding box     │
                        │   - Coordinate axes     │
                        │   - Rendered overlay    │
                        │                         │
                        │ • Pose file save        │
                        │   debug_dir/ob_in_cam/  │
                        │   {frame_id}.txt        │
                        └─────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              DATA FLOW DETAILS                                 │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    RGB-D Input ─┬─→ Preprocessing ─┬─→ Initial Pose Generation
                 │                  │
                 │                  ├─→ Pose Refinement Loop
                 │                  │   ├─ Crop & Resize
                 │                  │   ├─ Synthetic Rendering
                 │                  │   ├─ RefineNet Forward
                 │                  │   └─ Pose Update
                 │                  │
                 └─→ Score Network ──┼─→ Best Pose Selection
                                    │
    3D Model ────┬─→ Mesh Processing ┼─→ Synthetic Rendering
                 │                  │
                 └─→ Rotation Grid ──┘
    
    Camera K ────┬─→ XYZ Mapping ────┬─→ Network Input
                 │                   │
                 └─→ Projection ──────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                            NETWORK ARCHITECTURES                               │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    RefineNet:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Input A   │    │   Input B   │    │             │
    │ (B,6,224,224)│    │ (B,6,224,224)│    │             │
    └─────────────┘    └─────────────┘    │             │
           │                  │           │             │
           └──────┬───────────┘           │             │
                  │                       │             │
                  ▼                       │             │
        ┌─────────────────┐               │             │
        │   EncoderA      │               │             │
        │ • Conv7×7,s=2   │               │             │
        │ • Conv3×3,s=2   │               │ Transformer │
        │ • ResBlock×2    │               │ Encoder     │
        └─────────────────┘               │ Layer       │
                  │                       │             │
                  ▼                       │             │
        ┌─────────────────┐               │             │
        │   EncoderAB     │               │             │
        │ • ResBlock×2    │               │             │
        │ • Conv3×3,s=2   │               │             │
        │ • ResBlock×2    │               │             │
        └─────────────────┘               │             │
                  │                       │             │
                  ▼                       │             │
        ┌─────────────────┐               │             │
        │ Positional      │               │             │
        │ Embedding       │               │             │
        └─────────────────┘               │             │
                  │                       │             │
                  ├─────────────────────────┼─────────────┘
                  │                       │
                  ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │   Trans Head    │     │   Rot Head      │
        │ • Transformer   │     │ • Transformer   │
        │ • Linear(3)     │     │ • Linear(3/6)   │
        └─────────────────┘     └─────────────────┘
                  │                       │
                  ▼                       ▼
            trans_delta              rot_delta
              (B, 3)                 (B, 3/6)
    
    ScoreNet:
    ┌─────────────┐    ┌─────────────┐
    │   Input A   │    │   Input B   │
    │(B*L,6,224,224)│   │(B*L,6,224,224)│
    └─────────────┘    └─────────────┘
           │                  │
           └──────┬───────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   EncoderA      │
        │ • Same as       │
        │   RefineNet     │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   EncoderAB     │
        │ • Same as       │
        │   RefineNet     │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Self-Attention  │
        │ • Multi-head    │
        │ • 4 heads       │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Cross-Attention │
        │ • Between       │
        │   candidates    │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   Linear(1)     │
        │   Score Head    │
        └─────────────────┘
                  │
                  ▼
               scores
              (B, L)
    
    ```
    
    1. **First Frame**: 초기 자세 추정 (Rotation Grid + Refinement + Scoring)
    2. **Tracking**: 이전 자세 기반 정제만 수행 (빠른 처리)
    3. **Two Networks**: RefineNet (자세 정제) + ScoreNet (자세 평가)
    4. **Synthetic Rendering**: 3D 모델로 가상 이미지 생성하여 실제 이미지와 비교
    5. **Iterative Refinement**: 점진적 자세 개선을 통한 정확도 향상
    
    ## 📊 FoundationPose 모델 Input/Output 상세 분석
    
    ### 🎯 모델 개요
    
    FoundationPose는 **6D 객체 자세 추정 및 추적**을 위한 통합 foundation 모델로, 두 가지 주요 네트워크로 구성됩니다:
    
    1. **ScoreNet**: 자세 후보들의 점수를 매기는 네트워크
    2. **RefineNet**: 자세를 refine하는 네트워크
    
    ---
    
    ## 🔍 전체 시스템 Input
    
    ### **메인 입력 데이터**
    
    ```python
    # 기본 입력 (run_demo.py 기준)
    - RGB 이미지: (H, W, 3) np.array, uint8
    - Depth 이미지: (H, W) np.array, float32 
    - Camera 내부 파라미터 K: (3, 3) np.array, float32
    - Object mask: (H, W) np.array, bool (첫 번째 프레임용)
    - 3D 모델: trimesh object (vertices, normals)
    
    ```
    
    ### **데이터 리더에서 제공하는 형식**
    
    ```python
    class YcbineoatReader:
        def get_color(i):    # RGB: (H, W, 3) uint8
        def get_depth(i):    # Depth: (H, W) float32 (mm 단위를 m로 변환)
        def get_mask(i):     # Mask: (H, W) bool
        def get_xyz_map(i):  # XYZ: (H, W, 3) 3D 좌표
    
    ```
    
    ---
    
    ## 🧠 ScoreNet Input/Output
    
    ### **Input**
    
    ```python
    # ScoreNetMultiPair.forward()
    A: (B*L, C, H, W)  # 렌더링된 이미지 (RGB + XYZ)
    B: (B*L, C, H, W)  # 실제 관측 이미지 (RGB + XYZ)
    L: int             # 자세 후보 개수
    
    # 채널 구성
    C = 4:  RGB(3) + Depth(1)
    C = 6:  RGB(3) + XYZ(3)  # xyz_map 사용시
    C = 7:  RGB(3) + XYZ(3) + Normal(3)  # normal 사용시
    
    ```
    
    ### **Preprocessing**
    
    ```python
    # 크롭 및 리사이징
    render_size = cfg['input_resize']  # 예: [224, 224]
    crop_ratio = 1.2  # 3D 바운딩 박스 기준으로 크롭
    tf_to_crops: (B, 3, 3)  # 크롭 변환 행렬
    
    # 데이터 정규화
    rgbA, rgbB: [0, 255] → [0, 1]
    depth: meter 단위 유지
    xyz_map: 카메라 좌표계 3D 점들
    
    ```
    
    ### **Output**
    
    ```python
    output = {
        'score_logit': (B, L)  # 각 자세 후보의 점수 로짓
    }
    
    ```
    
    ---
    
    ## 🎯 RefineNet Input/Output
    
    ### **Input**
    
    ```python
    # RefineNet.forward()
    A: (B, C, H, W)  # 렌더링된 이미지
    B: (B, C, H, W)  # 실제 관측 이미지
    
    # 채널 구성 동일 (ScoreNet과 같음)
    C = 4 또는 6 또는 7
    
    ```
    
    ### **Output**
    
    ```python
    output = {
        'trans': (B, 3),    # 평행이동 델타 (x, y, z)
        'rot': (B, 3 또는 6) # 회전 표현
    }
    
    # 회전 표현 옵션
    rot_rep = 'axis_angle': (B, 3)  # 축-각도 표현
    rot_rep = '6d': (B, 6)          # 6D 회전 표현
    
    ```
    
    ### **변환 처리**
    
    ```python
    # 평행이동 정규화
    trans_delta = tanh(output["trans"]) * trans_normalizer
    # 회전 정규화
    rot_delta = tanh(output["rot"]) * rot_normalizer
    rot_matrix = so3_exp_map(rot_delta)  # 축-각도 → 회전행렬
    
    ```
    
    ---
    
    ## 🔄 데이터 플로우
    
    ### **1. 초기 자세 추정 (register)**
    
    ```python
    # 1단계: 후보 자세 생성
    poses = generate_random_pose_hypo()  # (N, 4, 4) 후보 자세들
    
    # 2단계: RefineNet으로 자세 정제
    refined_poses = refiner.predict(
        rgb=color,           # (H, W, 3)
        depth=depth,         # (H, W)
        K=K,                 # (3, 3)
        ob_in_cams=poses,    # (N, 4, 4)
        iteration=5          # 정제 반복 횟수
    )
    
    # 3단계: ScoreNet으로 점수 매기기
    scores = scorer.predict(
        rgb=color,
        depth=depth,
        K=K,
        ob_in_cams=refined_poses
    )
    
    # 4단계: 최고 점수 자세 선택
    best_pose = refined_poses[scores.argmax()]
    
    ```
    
    ### **2. 추적 (track_one)**
    
    ```python
    # 이전 자세를 시작점으로 정제만 수행
    pose = refiner.predict(
        rgb=color,
        depth=depth,
        K=K,
        ob_in_cams=previous_pose.reshape(1,4,4),
        iteration=2
    )
    
    ```
    
    ---
    
    ## 📐 배치 처리 형식
    
    ### **BatchPoseData 구조**
    
    ```python
    @dataclass
    class BatchPoseData:
        rgbAs: (B, 3, H, W)      # 렌더링 이미지
        rgbBs: (B, 3, H, W)      # 관측 이미지
        depthAs: (B, 1, H, W)    # 렌더링 깊이
        depthBs: (B, 1, H, W)    # 관측 깊이
        xyz_mapAs: (B, 3, H, W)  # 렌더링 3D 좌표
        xyz_mapBs: (B, 3, H, W)  # 관측 3D 좌표
        normalAs: (B, 3, H, W)   # 렌더링 법선 (옵션)
        normalBs: (B, 3, H, W)   # 관측 법선 (옵션)
        poseA: (B, 4, 4)         # 자세 정보
        tf_to_crops: (B, 3, 3)   # 크롭 변환
        Ks: (B, 3, 3)           # 카메라 파라미터
        mesh_diameters: (B,)     # 메시 직경
    
    ```
    
    ---
    
    ## ⚙️ 핵심 파라미터
    
    ### **네트워크 설정**
    
    ```python
    # ScoreNet
    embed_dim = 512
    num_heads = 4
    max_sequence_length = 400
    
    # RefineNet
    embed_dim = 512
    num_heads = 4
    trans_normalizer = 0.01   # 평행이동 정규화
    rot_normalizer = 0.1      # 회전 정규화
    
    # 입력 크기
    input_resize = [224, 224]  # 네트워크 입력 해상도
    crop_ratio = 1.2          # 크롭 비율
    
    ```
    
    ### **출력 자세 형식**
    
    ```python
    # 최종 출력
    pose: (4, 4) np.array     # 객체에서 카메라로의 변환 행렬
    pose[:3, :3]              # 회전 행렬 (3x3)
    pose[:3, 3]               # 평행이동 벡터 (3,)
    
    ```
    

FoundationPose 모델의 전체 파이프라인을 텍스트로 그려드리겠습니다:

- 빠르게 스킵
    
    ```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                           FOUNDATIONPOSE PIPELINE                              │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   INPUT DATA    │    │   3D MODEL      │    │   CAMERA INFO   │
    │                 │    │                 │    │                 │
    │ • RGB Image     │    │ • Mesh (.obj)   │    │ • Intrinsic K   │
    │   (H, W, 3)     │    │ • Vertices      │    │   (3, 3)        │
    │ • Depth Image   │    │ • Normals       │    │ • Resolution    │
    │   (H, W)        │    │ • Textures      │    │   (H, W)        │
    │ • Object Mask   │    │                 │    │                 │
    │   (H, W) [초기] │    │                 │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │    PREPROCESSING        │
                        │                         │
                        │ • Depth Filtering       │
                        │   - Erosion (radius=2)  │
                        │   - Bilateral Filter    │
                        │ • XYZ Map Generation    │
                        │ • Mesh Processing       │
                        │   - Center to Origin    │
                        │   - Compute Diameter    │
                        │   - Voxel Downsample    │
                        └─────────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────────────────┐
                │                POSE ESTIMATION                     │
                │              (First Frame Only)                    │
                └────────────────────────────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │  INITIAL POSE GUESS     │
                        │                         │
                        │ 1. Rotation Grid        │
                        │    • Icosphere Views    │
                        │    • Inplane Rotations  │
                        │    • Symmetry Cluster   │
                        │      → (N_views, 4, 4)  │
                        │                         │
                        │ 2. Translation Guess    │
                        │    • Mask Center (u,v)  │
                        │    • Median Depth (z)   │
                        │    • K^-1 * [u,v,1] * z │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     POSE REFINER        │
                        │    (Iteration × 5)      │
                        │                         │
                        │ For each pose candidate:│
                        │                         │
                        │ Input Preparation:      │
                        │ • Crop around 3D bbox   │
                        │ • Resize to 224×224     │
                        │ • Render synthetic view │
                        │                         │
                        │ RefineNet Forward:      │
                        │ A: Render (B,6,224,224) │
                        │ B: Real   (B,6,224,224) │
                        │ ├─ RGB (3 channels)     │
                        │ └─ XYZ (3 channels)     │
                        │                         │
                        │ Output:                 │
                        │ • trans_delta: (B, 3)   │
                        │ • rot_delta: (B, 3)     │
                        │                         │
                        │ Update:                 │
                        │ pose_new = pose_old +   │
                        │           delta_pose    │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     SCORE NETWORK       │
                        │                         │
                        │ For all refined poses:  │
                        │                         │
                        │ Input Preparation:      │
                        │ • Same cropping process │
                        │ • Batch all candidates  │
                        │                         │
                        │ ScoreNet Forward:       │
                        │ A: Render (B*L,6,224,224)│
                        │ B: Real   (B*L,6,224,224)│
                        │ L: number of candidates │
                        │                         │
                        │ Multi-head Attention:   │
                        │ • Self-attention on     │
                        │   image features        │
                        │ • Cross-attention       │
                        │   between candidates    │
                        │                         │
                        │ Output:                 │
                        │ • scores: (B, L)        │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │    BEST POSE SELECTION  │
                        │                         │
                        │ • Sort by score         │
                        │ • Select highest        │
                        │ • Store for tracking    │
                        │                         │
                        │ best_pose = poses[      │
                        │   scores.argmax()]      │
                        └─────────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────────────────┐
                │                    TRACKING                        │
                │               (Subsequent Frames)                  │
                └────────────────────────────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   POSE REFINEMENT       │
                        │    (Iteration × 2)      │
                        │                         │
                        │ Input:                  │
                        │ • Previous frame pose   │
                        │ • Current RGB-D         │
                        │                         │
                        │ Process:                │
                        │ • Single pose input     │
                        │ • Same RefineNet        │
                        │ • Fewer iterations      │
                        │                         │
                        │ Output:                 │
                        │ • Refined pose          │
                        │ • Update pose_last      │
                        └─────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     FINAL OUTPUT        │
                        │                         │
                        │ • 6DOF Pose Matrix      │
                        │   (4, 4) transformation │
                        │                         │
                        │ • Visualization         │
                        │   - 3D bounding box     │
                        │   - Coordinate axes     │
                        │   - Rendered overlay    │
                        │                         │
                        │ • Pose file save        │
                        │   debug_dir/ob_in_cam/  │
                        │   {frame_id}.txt        │
                        └─────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              DATA FLOW DETAILS                                 │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    RGB-D Input ─┬─→ Preprocessing ─┬─→ Initial Pose Generation
                 │                  │
                 │                  ├─→ Pose Refinement Loop
                 │                  │   ├─ Crop & Resize
                 │                  │   ├─ Synthetic Rendering
                 │                  │   ├─ RefineNet Forward
                 │                  │   └─ Pose Update
                 │                  │
                 └─→ Score Network ──┼─→ Best Pose Selection
                                    │
    3D Model ────┬─→ Mesh Processing ┼─→ Synthetic Rendering
                 │                  │
                 └─→ Rotation Grid ──┘
    
    Camera K ────┬─→ XYZ Mapping ────┬─→ Network Input
                 │                   │
                 └─→ Projection ──────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                            NETWORK ARCHITECTURES                               │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    RefineNet:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Input A   │    │   Input B   │    │             │
    │ (B,6,224,224)│    │ (B,6,224,224)│    │             │
    └─────────────┘    └─────────────┘    │             │
           │                  │           │             │
           └──────┬───────────┘           │             │
                  │                       │             │
                  ▼                       │             │
        ┌─────────────────┐               │             │
        │   EncoderA      │               │             │
        │ • Conv7×7,s=2   │               │             │
        │ • Conv3×3,s=2   │               │ Transformer │
        │ • ResBlock×2    │               │ Encoder     │
        └─────────────────┘               │ Layer       │
                  │                       │             │
                  ▼                       │             │
        ┌─────────────────┐               │             │
        │   EncoderAB     │               │             │
        │ • ResBlock×2    │               │             │
        │ • Conv3×3,s=2   │               │             │
        │ • ResBlock×2    │               │             │
        └─────────────────┘               │             │
                  │                       │             │
                  ▼                       │             │
        ┌─────────────────┐               │             │
        │ Positional      │               │             │
        │ Embedding       │               │             │
        └─────────────────┘               │             │
                  │                       │             │
                  ├─────────────────────────┼─────────────┘
                  │                       │
                  ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │   Trans Head    │     │   Rot Head      │
        │ • Transformer   │     │ • Transformer   │
        │ • Linear(3)     │     │ • Linear(3/6)   │
        └─────────────────┘     └─────────────────┘
                  │                       │
                  ▼                       ▼
            trans_delta              rot_delta
              (B, 3)                 (B, 3/6)
    
    ScoreNet:
    ┌─────────────┐    ┌─────────────┐
    │   Input A   │    │   Input B   │
    │(B*L,6,224,224)│   │(B*L,6,224,224)│
    └─────────────┘    └─────────────┘
           │                  │
           └──────┬───────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   EncoderA      │
        │ • Same as       │
        │   RefineNet     │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   EncoderAB     │
        │ • Same as       │
        │   RefineNet     │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Self-Attention  │
        │ • Multi-head    │
        │ • 4 heads       │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Cross-Attention │
        │ • Between       │
        │   candidates    │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   Linear(1)     │
        │   Score Head    │
        └─────────────────┘
                  │
                  ▼
               scores
              (B, L)
    
    ```
    

Realsense 435di →  이제 카메라 자체의 왜곡을 생각을 해보자

1. RGB원리는 OK + depth 정보가 정확하지 못함.
    1. 특히 RGB + D로 Yolo8v 2d image를 가지고 학습을 시켰는데, 
    2. 3d 포인트 클라우드를 추출해야한다?? 일단 정확도가 나올리가 없음. 왜냐하면 모델이 3차원 공간을 이해할리가 만무함. 아마도 train한 코드를 보면 강화학습도 아니고 cnn기반이니까 데이터 라벨링으로 정답을 찾는거잖음 그렇다면. 카메라의 3d인식 정보가 문제가 있다는 거 아닐까?
    
    ## 리얼센스 카메라 D435i
    
    ![image.png](CV/image.png)
    

일단 

![image.png](CV/image%201.png)

- 파이썬 코드
    
    import cv2
    import time
    import math
    import numpy as np
    import pyrealsense2 as rs
    from ultralytics import YOLO
    
    ## Config
    
    ```python
    WEIGHTS = "../train_results/exp1/weights/best.pt" #2d yolo 가중치
    CONF_TH = 0.5 #Confidence Threshold : 탐지된 객체가 예측된 확률
    IOU_TH  = 0.5 #50% 이상 겹치는 OBB(포인트클라우드)는 중복으로 판단하고 제거
    # 모델의 세팅
    
    COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30 #카메라 RGB 스트리밍 해상도
    DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30 #카메라 Depth 스트리밍 해상도
    
    # 후처리에서 처리하는거
    SAMPLE_STRIDE = 2 #포인트클라우드 샘플링 시 stride :2개씩 한 점으로 여긴다는 이야기
    Z_MIN, Z_MAX = 0.1, 2.0 #0.1m ~ 2.0m 만 유효 데이터로 처리
    FONT = cv2.FONT_HERSHEY_SIMPLEX #bounding box나 텍스트 라벨링 시 활용
    
    # Extrinsic(외부 환경ㅊ) 파라미터
    RX_DEG, RY_DEG, RZ_DEG = -180.0, 0.0, -80.0 #회전 기준 각도
    TX, TY, TZ = 0.20, 0.0, 0.50 #평행 이동 단위 
    ORDER = "XYZ" #회전 순서
    ```
    
    ## util모음
    
    ```python
    # --------------------
    # Small math helpers
    # --------------------
    def Rx(t): #x축 회전 행렬
        c, s = math.cos(t), math.sin(t)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)
    
    def Ry(t): #y축 회전 행렬
        c, s = math.cos(t), math.sin(t)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)
    
    def Rz(t): #z축 회전 행렬
        c, s = math.cos(t), math.sin(t)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    
    #외부 환경 파라미터 -> 변환행렬 H를 만드는 (회전 + 평행이동)
    def H_from_axis_angles(tx, ty, tz, rx, ry, rz, order="XYZ"):
        R_map = {"X": Rx(rx), "Y": Ry(ry), "Z": Rz(rz)}
        R = np.eye(3)
        for ax in order:
            R = R @ R_map[ax]
        H = np.eye(4)
        H[:3,:3] = R
        H[:3, 3] = [tx, ty, tz]
        return H
    
    # 오른쪽 손임을 미리 세팅
    def ensure_right_handed(R):
        R = np.asarray(R, float).copy()
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0
        return R
    ```
    
    # 2D↔3D 변환
    
    ```python
    # --------------------
    # Camera projections
    # --------------------
    def precompute_xy_maps(intr, H, W): 
    #촬영한 2차원 좌표를 depth와 곱하면 3D 좌표를 얻을 수 있게 미리 계산 -> 정확한지?
        js = np.arange(W, dtype=np.float32)
        is_ = np.arange(H, dtype=np.float32)
        gy, gx = np.meshgrid(is_, js, indexing="ij")
        x_map = (gx - intr.ppx) / intr.fx
        y_map = (gy - intr.ppy) / intr.fy
        return x_map, y_map
    
    def project_points_intr(intr, pts3d):
    #3d -> 2d의 정보 추출 픽셀 좌표 추출
        pts3d = np.asarray(pts3d, dtype=np.float32)
        Z = pts3d[:, 2]
        valid = Z > 1e-6
        uv = np.full((pts3d.shape[0], 2), np.nan, np.float32)
        uv[valid, 0] = intr.fx * (pts3d[valid, 0] / Z[valid]) + intr.ppx
        uv[valid, 1] = intr.fy * (pts3d[valid, 1] / Z[valid]) + intr.ppy
        return uv, valid
    
    ```
    

## 중요한, depth→ point cloud 추출

```python
# --------------------
# Depth → 3D points (mask-based)
# --------------------
def mask_to_points3d(depth_frame, mask, depth_scale, intr, x_map, y_map,
                     sample_stride=1, z_min=0.0, z_max=10.0, erosion=3):
     """
     depth_frame: RealSense 등에서 받은 Depth 이미지 (16bit depth)
     mask: Segmentation이나 Detection 결과 마스크 (ROI 정의) #yolo에서 뽑은거
     depth_scale: 깊이 데이터 단위 변환 계수 (16bit -> 미터)
     intr: 카메라 intrinsic(내부 데이터) 파라미터 (fx, fy, ppx, ppy)
     x_map, y_map: 미리 계산된 정규화 좌표 맵 (precompute_xy_maps)
     sample_stride: 샘플링 stride (계산량 줄이기용) ##점 갯수 줄이기
     z_min, z_max: 깊이 필터링 범위 (m 단위) 0.2~2.0으로 알면 됨
     erosion: 마스크 침식 커널 크기 (노이즈 제거용) #작은 노이즈 점 제거 경계선 다듬기
     """
    depth_u16 = np.asanyarray(depth_frame.get_data()) 
    H, W = depth_u16.shape[:2]
    #넘파이 변환
    
    # 마스크 크기가 다르면 depth와 맞게 리사이즈 (마스크 크기 다르면 어떻게 해야할까...흠)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    
    # cv2.erode => 경계부를 줄여서 노이즈 제거  
    if erosion > 0:
        k = np.ones((erosion*2+1, erosion*2+1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k, 1)
        
        
    # 마스크 내부(True)
    m = (mask > 0)
    
    
    # 샘플링 stride 세팅값이 있을 때
    if sample_stride > 1:
        m = m[::sample_stride, ::sample_stride]
        Zf = depth_u16.astype(np.float32) * depth_scale
        Z = Zf[::sample_stride, ::sample_stride]
        X = (x_map * Zf)[::sample_stride, ::sample_stride]
        Y = (y_map * Zf)[::sample_stride, ::sample_stride]
    
    #그냥 데이터 그대로    
    else:
        Z = depth_u16.astype(np.float32) * depth_scale
        X = x_map * Z
        Y = y_map * Z
        
    if not np.any(m):
        return None
    
    # 마스크 내부 픽셀만 추출 : 깊이>0 && 유효 && z_min~z_max => valid
    Xv, Yv, Zv = X[m], Y[m], Z[m]
    valid = (Zv > 0) & np.isfinite(Zv) & (Zv >= z_min) & (Zv <= z_max)
    
    if not np.any(valid):
        return None
        
    #최종 3d 검출
    pts = np.stack([Xv[valid], Yv[valid], Zv[valid]], axis=1)
    return pts if pts.shape[0] >= 30 else None

```

## PCA를 통해 3d 박스 구하기

```python
def pca_obb_3d(points_xyz):
    # 입력: points_xyz (N×3) 포인트 클라우드 좌표들

    pts = points_xyz.astype(np.float32)  
    # 포인트 클라우드 (float32 형변환)

    mean = pts.mean(axis=0)  
    # 포인트 클라우드의 평균 좌표 

    C = np.cov((pts - mean), rowvar=False)  
    # 공분산 행렬 (점들의 분포의 방향성 계산)

    vals, vecs = np.linalg.eigh(C)  
    # 고유값(vals), 고유벡터(vecs) 추출 (PCA)

    order = np.argsort(vals)[::-1]  
    # 고유값을 큰 값 -> 작은 값 순서대로 정렬 (분산이 큰 축부터)

    eigvals_desc = vals[order]  
    # 정렬된 고유값 (축별 분산 크기, 안정성 지표)

    axes = vecs[:, order]  
    # 정렬된 고유벡터 (OBB의 주축 방향)

    axes = axes / (np.linalg.norm(axes, axis=0, keepdims=True) + 1e-9)  
    # 각 축 벡터 정규화 (길이 1로)

    proj = (pts - mean) @ axes  
    # 모든 점들을 PCA 좌표계(주축 기준 좌표계)로 투영

    mins, maxs = proj.min(axis=0), proj.max(axis=0)  
    # 각 주축 방향으로 최소/최대 값 (분포 범위)

    c_local = (mins + maxs) * 0.5  
    # PCA 좌표계에서 OBB 중심 (local frame 기준)

    half = (maxs - mins) * 0.5  
    # OBB의 반 길이 (X, Y, Z 방향 반쪽 길이)

    center = mean + axes @ c_local  
    # 원래 좌표계에서의 OBB 중심 좌표

    corners = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            for s3 in (+1, -1):
                # 8개 꼭짓점 좌표 계산
                corners.append(
                    center
                    + s1 * half[0] * axes[:, 0]
                    + s2 * half[1] * axes[:, 1]
                    + s3 * half[2] * axes[:, 2]
                )
    corners = np.stack(corners, axis=0)  
    # (8×3) 꼭짓점 좌표 배열

    lengths = 2.0 * half  
    # OBB의 가로/세로/높이 길이

    return center, axes, lengths, corners, eigvals_desc
    # center: 박스 중심 (3,)
    # axes: 박스 축 방향 (3×3)
    # lengths: 박스 크기 (3,)
    # corners: 꼭짓점 좌표 (8×3)
    # eigvals_desc: 고유값 (분산 크기, 안정성 확인용)
```

## PCA 포즈 필터 : 이상한 포인트 클라우드 와도 무시

```python
# --------------------
# PCA Pose Filter for stability
# --------------------
class PCAPoseFilter:
    def __init__(self, ratio_thresh=1.5, keep_last_when_unstable=True):
        """
        ratio_thresh: r1 = λ1/λ2, r2 = λ2/λ3 
        둘 다 기준값 (ratio thresh)보다 커야 안정 으로 판단
        keep_last_when_unstable: 불안정하면 직전 포즈를 그대로 반환
        """
        self.ratio_thresh = float(ratio_thresh)#stable지표
        self.keep_last = bool(keep_last_when_unstable)
        self.prev_center = None
        self.prev_axes = None
        self.prev_lengths = None
        self.prev_corners = None
	      

    @staticmethod
    def _ensure_right_handed(R): #오른쪽이니까 보정
        R = np.asarray(R, float).copy()
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0
        return R

    def _align_to_prev(self, axes_new):
        """축 방향 부호를 이전 프레임과 정렬 (u vs -u 뒤집힘 방지)"""
        if self.prev_axes is None:
            return axes_new
        axes = axes_new.copy()
        dots = np.sum(axes * self.prev_axes, axis=0)  # 열벡터별 내적
        flip = dots < 0
        axes[:, flip] *= -1
        return axes

    def _is_stable(self, eigvals_desc):
        # eigvals_desc: 내림차순 [λ1, λ2, λ3]
        if len(eigvals_desc) < 3:
            return False
        l1, l2, l3 = [max(1e-12, float(v)) for v in eigvals_desc]
        r1 = l1 / l2
        r2 = l2 / l3
        return (r1 > self.ratio_thresh) and (r2 > self.ratio_thresh) 
        
    
    def update(self, center, axes, lengths, corners, eigvals_desc):
        """
        안정성 검사 후 최종 포즈를 반환.
        - 안정: 부호 정렬 + 오른손계 보정 후 갱신/반환
        - 불안정: 이전 포즈 유지(있으면) / 없으면 현재라도 반환
        """
        stable = self._is_stable(eigvals_desc)

        if not stable and self.keep_last and (self.prev_axes is not None):
            # 직전 포즈 그대로 사용
            return self.prev_center, self.prev_axes, self.prev_lengths, self.prev_corners, False

        # 안정적이거나(업데이트), 직전 포즈가 없을 때는 새 포즈 정리
        axes = self._align_to_prev(axes)
        axes = self._ensure_right_handed(axes)

        # 상태 저장
        self.prev_center  = center
        self.prev_axes    = axes
        self.prev_lengths = lengths
        self.prev_corners = corners

    return center, axes, lengths, corners, True

```

## RANSAC 알고리즘 → 노이즈가 많은 데이터에서 최빈값 평면 찾기

```python
def fit_plane_ransac(pts, iters=100, tau=0.01):
    """
    입력:
	    pts: N×3 포인트 클라우드 점 집합
	    iters: 반복 횟수 (무작위 샘플링 횟수)
	    tau: inlier 판정 거리 한계값
	    
    출력:
	    n: 단위 법선 벡터 (평면 방향, z값 보정)
	    d: 평면 방정식의 offset
    """
    N = pts.shape[0]
    
    if N < 50: #50미만 나가라
        return None, None
    best_inl, best = 0, (None, None)
    rng = np.random.default_rng()
    
    #무작위로 3개 점 선택 (a, b, c)으로 평면의 방정식의 평면벡터 와구자와구자 구하기
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        a,b,c = pts[idx]
        n = np.cross(b-a, c-a) #평면 두 벡터의 외적하며 나오죠잉
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9: 
            continue
         
        # 거리가 tau 이하인 점 = inlier, 이상인 점 outlier
        n = n / n_norm
        d = -np.dot(n, a)
        dist = np.abs(pts @ n + d)
        inl = np.count_nonzero(dist < tau)
        if inl > best_inl: #inlier 개수가 많은 평면 선택 (최대값 찾기)
            best_inl = inl; best = (n, d)
    
    if best[0] is None:
        return None, None
  
    n = best[0] 
    if n[2] < 0: n = -n #법선 벡터가 아래(-z)를 향하면 반대로 구하기
    return n / (np.linalg.norm(n)+1e-9), best[1]
  
	 
```

## 3D 회전 수학 유틸리티 함수들

```python
# --------------------
# Robust pose stabilization (roll/pitch lock + temporal smoothing)
# --------------------
def so3_log(R):
    """Matrix log for SO(3) -> axis*angle (vector)"""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * theta

def so3_exp(w):
    """Axis-angle vector -> SO(3) using Rodrigues 로드리게스 행렬 적용 툴"""
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K@K)

def slerp_SO3(R0, R1, alpha):
    """Geodesic interpolation on SO(3). : 두 회전 사이를 매끄럽게 만든다"""
    dR = R0.T @ R1
    w = so3_log(dR)
    return R0 @ so3_exp(alpha * w)
```

## 평면 기반 깊이 채우기

```python
# --------------------
# Geometric depth filling (plane-based interpolation)
# --------------------
def fill_by_plane(depth_u16, mask, depth_scale, intr, x_map, y_map, pts3d, tau=0.01):
    """
    평면 기반 깊이 보간: 마스크 내부의 누락된 깊이값을 RANSAC 평면으로 채움
    테이블, 바닥 등 평면성이 있는 객체에 효과적 (근데 물체에는? 잘모르겠다)
    
    Args:
        depth_u16: 원본 깊이 이미지 (uint16)
        mask: 객체 마스크
        depth_scale: 깊이 스케일 팩터
        intr: 카메라 내부 파라미터
        x_map, y_map: 사전 계산된 좌표 맵
        pts3d: 3D 포인트 클라우드 (평면 피팅용)
        tau: RANSAC 임계값
    
    Returns:
        채워진 깊이 이미지 (uint16)
    """
    n, d = fit_plane_ransac(pts3d, tau=tau)
    if n is None: 
        return depth_u16
    
    H, W = depth_u16.shape
    # 카메라 좌표: X = x_map * Z, Y = y_map * Z
    # 평면 방정식: n * (Z * [x_map, y_map, 1]) + d = 0
    # => Z = -d / (n * [x_map, y_map, 1])
    Z = depth_u16.astype(np.float32) * depth_scale
    denom = (n[0] * x_map + n[1] * y_map + n[2])
    Z_plane = -d / (denom + 1e-9)
    
    # 마스크 내부에서 깊이가 없는(<=0) 픽셀에만 평면 깊이 적용
    m = (mask > 0) & (Z <= 0)
    Z[m] = np.clip(Z_plane[m], 0.0, 10.0)
    
    return (Z / depth_scale).astype(np.uint16)

```

## 

## **시간적 스무딩(회전=SLERP,  평행이동=EMA)**

## **roll/pitch 고정( 평면 결정 후 )**

```python
class PoseStabilizer:
    """Temporal smoothing + roll/pitch lock via plane normal or IMU gravity."""
    def __init__(self, alpha_R=0.25, alpha_t=0.3, use_plane_lock=True):
        self.R_prev = None
        self.t_prev = None
        self.alpha_R = alpha_R
        self.alpha_t = alpha_t
        self.use_plane_lock = use_plane_lock
        self.z_up_ref = None  # set from first reliable plane or external IMU

    def lock_roll_pitch(self, R_obj_cam, z_up):
        """회전은 없음을,,, 롤 피치 고정"""
        # build a camera/world frame where z = z_up
        z = z_up / (np.linalg.norm(z_up)+1e-9)
        # choose x as projection of current x onto plane, then y = z×x
        x_raw = R_obj_cam[:,0]
        x = x_raw - np.dot(x_raw, z)*z
        if np.linalg.norm(x) < 1e-6:
            return R_obj_cam
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R_locked = np.stack([x, y, z], axis=1)
        return R_locked

    def update(self, center3d, axes3, pts3d):
        # 1) 안정화 되면 평면 고정
        if self.use_plane_lock:
            n, _ = fit_plane_ransac(pts3d)
            if n is not None:
                self.z_up_ref = n if self.z_up_ref is None else 0.8*self.z_up_ref + 0.2*n
                axes3 = self.lock_roll_pitch(axes3, self.z_up_ref)

        # 2) temporal smoothing (SO(3) SLERP + EMA for translation)
        if self.R_prev is None:
            self.R_prev = axes3.copy()
            self.t_prev = center3d.copy()
            return center3d, axes3

        R_s = slerp_SO3(self.R_prev, axes3, self.alpha_R)
        t_s = (1.0 - self.alpha_t)*self.t_prev + self.alpha_t*center3d

        self.R_prev = R_s
        self.t_prev = t_s
        return t_s, R_s

```

# main 루프

## 초기화

```python
model = YOLO(WEIGHTS)      
names = model.names      

pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
prof = pipe.start(cfg)      # RealSense 카메라 동작 시작~

```

- YOLO 모델 준비
- RealSense color/depth 스트림 해상도 및 FPS 설정 후 파이프라인 실행

---

## 2. 보정과 3d→ 2d 맵 준비

```python
depth_sensor = prof.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()   # 깊이 단위 (mm→m)

align = rs.align(rs.stream.color)              # depth→color 정렬
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
x_map, y_map = precompute_xy_maps(intr, COLOR_H, COLOR_W)

```

깊이값 스케일링

color 기준으로 depth 정렬

Intrinsic(내부 파라미터) 추출, 포인트클라우드 변환용 맵(x_map,y_map) 준비

---

## 3. Extrinsic (기준 좌표계 변환) → 임시값이라 부정확할 것 같음

```python
H_BC = H_from_axis_angles(TX, TY, TZ,
                          math.radians(RX_DEG), math.radians(RY_DEG), math.radians(RZ_DEG),
                          order=ORDER)
R_BC = H_BC[:3,:3]

```

카메라 좌표계를 base/world 좌표계로 맞추는 extrinsic 변환 행렬

---

## 4. 필터/안정화 class 준비

```python
pca_filter = PCAPoseFilter(ratio_thresh=1.5, keep_last_when_unstable=True)
stabilizer = PoseStabilizer(alpha_R=0.25, alpha_t=0.3, use_plane_lock=True)

```

- `PCAPoseFilter`: PCA OBB 결과 안정성 필터 (eigenvalue 비율 확인)
- `PoseStabilizer`: roll/pitch 고정 + temporal smoothing

---

## 5. 메인 루프

```python
while True:
    frames = pipe.wait_for_frames()
    aligned = align.process(frames)
    d = aligned.get_depth_frame(); c = aligned.get_color_frame()
    if not d or not c: continue

```

- 프레임 수신 + depth/color 정렬

---

## 6. YOLO 추론

```python
color = np.asanyarray(c.get_data())
overlay = color.copy()

rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
r = model(rgb, conf=CONF_TH, iou=IOU_TH, verbose=False)[0]

boxes = ...
clses = ...
confs = ...
masks = ...

```

- YOLO 모델 실행
- bbox, 클래스, confidence, segmentation mask 추출

---

## 7. 객체 단위 처리

```python
for i in range(len(boxes)):
    # 마스크 있으면 사용, 없으면 bbox fallback
    pts3d = mask_to_points3d(d, mask, depth_scale, intr, x_map, y_map, ...)
    if pts3d is None: continue

    # PCA OBB 추정
    center3d, axes3, lens3, corners3d, eigvals_desc = pca_obb_3d(pts3d)

    # PCA 필터 안정화
    center3d, axes3, lens3, corners3d, is_stable = pca_filter.update(...)

    # 추가 안정화 (roll/pitch lock + temporal smoothing)
    center3d, axes3 = stabilizer.update(center3d, axes3, pts3d)

    # 시각화 (3D 박스 + 축)
    draw_obb3d_on_image(overlay, intr, corners3d)
    draw_axes3d(overlay, intr, center3d, axes3, lens3)

    # 레이블 (클래스명, conf, 깊이, 안정성)
    uv_c, ok = project_points_intr(intr, center3d.reshape(1,3))
    if ok[0]:
        cx, cy = ...
        stability_text = "STABLE"/"UNSTABLE"
        cv2.putText(...)

        # Yaw(방향) 표시
        objX = (H_BC[:3,:3] @ axes3)[:,0]
        yaw_deg = atan2(objX[1], objX[0])
        cv2.putText(...)

```

---

## 8. FPS 출력 & 윈도우 표시

```python
n += 1
if n >= 10:
    now = time.time(); fps = n/(now-t0); t0=now; n=0
if fps is not None:
    cv2.putText(overlay, f"FPS: {fps:.1f}", ...)

cv2.imshow("RealSense YOLO (Core)", overlay)
if (cv2.waitKey(1) & 0xFF) == ord('q'):
    break

```

- FPS 계산
- OpenCV 윈도우로 결과 시각화

---

## 9. 종료 처리

```python
finally:
    pipe.stop()
    cv2.destroyAllWindows()

```

- 파이프라인 종료 및 리소스 정리
    
    [3d_obb.py](CV/3d_obb.py)