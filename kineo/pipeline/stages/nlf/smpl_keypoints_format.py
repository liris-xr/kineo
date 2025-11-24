from kineo.annotations import KeypointsFormat

NLF_SMPLX_KEYPOINTS_FORMAT = KeypointsFormat(
    name="nlf_smplx",
    n_keypoints=55 + 1024,
    keypoints_names=[
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "left_eye",
        "right_eye",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ]
    + [f"vertex_{i}" for i in range(1024)],
    keypoints_connectivity=[
        # Left leg
        (0, 1),
        (1, 4),
        (4, 7),
        (7, 10),
        # Right leg
        (0, 2),
        (2, 5),
        (5, 8),
        (8, 11),
        # Spine
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 15),
        (15, 22),
        (15, 23),
        (15, 24),
        # Left arm
        (9, 13),
        (13, 16),
        (16, 18),
        (18, 20),
        (20, 25), # left_wrist -> left_index1
        (25, 26), # left_index1 -> left_index2
        (26, 27), # left_index2 -> left_index3
        (20, 28), # left_wrist -> left_middle1
        (28, 29), # left_middle1 -> left_middle2
        (29, 30), # left_middle2 -> left_middle3
        (20, 31), # left_wrist -> left_pinky1
        (31, 32), # left_pinky1 -> left_pinky2
        (32, 33), # left_pinky2 -> left_pinky3
        (20, 34), # left_wrist -> left_ring1
        (34, 35), # left_ring1 -> left_ring2
        (35, 36), # left_ring2 -> left_ring3
        (20, 37), # left_wrist -> left_thumb1
        (37, 38), # left_thumb1 -> left_thumb2
        (38, 39), # left_thumb2 -> left_thumb3
        # Right arm
        (9, 14),
        (14, 17),
        (17, 19),
        (19, 21),
        (21, 40), # right_wrist -> right_index1
        (40, 41), # right_index1 -> right_index2
        (41, 42), # right_index2 -> right_index3
        (21, 43), # right_wrist -> right_middle1
        (43, 44), # right_middle1 -> right_middle2
        (44, 45), # right_middle2 -> right_middle3
        (21, 46), # right_wrist -> right_pinky1
        (46, 47), # right_pinky1 -> right_pinky2
        (47, 48), # right_pinky2 -> right_pinky3
        (21, 49), # right_wrist -> right_ring1
        (49, 50), # right_ring1 -> right_ring2
        (50, 51), # right_ring2 -> right_ring3
        (21, 52), # right_wrist -> right_thumb1
        (52, 53), # right_thumb1 -> right_thumb2
        (53, 54), # right_thumb2 -> right_thumb3
    ],
)

NLF_SMPL_KEYPOINTS_FORMAT = KeypointsFormat(
    name="nlf_smpl",
    n_keypoints=24 + 1024,
    keypoints_names=[
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ]
    + [f"vertex_{i}" for i in range(1024)],
    keypoints_connectivity=[
        # Left leg
        (0, 1),
        (1, 4),
        (4, 7),
        (7, 10),
        # Right leg
        (0, 2),
        (2, 5),
        (5, 8),
        (8, 11),
        # Spine
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 15),
        # Left arm
        (9, 13),
        (13, 16),
        (16, 18),
        (18, 20),
        (20, 22),
        # Right arm
        (9, 14),
        (14, 17),
        (17, 19),
        (19, 21),
        (21, 23),
    ],
)
