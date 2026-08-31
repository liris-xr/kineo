from kineo.annotations import KeypointsFormat

SMPL_24_KEYPOINTS_FORMAT = KeypointsFormat(
    name="smpl_24",
    n_keypoints=24,
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
    ],
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

# Reference for the mapping https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/issues/16#issuecomment-3160388104
COCO_19_KEYPOINTS_FORMAT = KeypointsFormat(
    name="coco_19",
    n_keypoints=19,
    keypoints_names=[
        "neck",
        "nose",
        "pelvis",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_eye",
        "left_ear",
        "right_eye",
        "right_ear",
    ],
    keypoints_connectivity=[
        (0, 1),  # neck → nose
        (0, 3),  # neck → left_shoulder
        (0, 9),  # neck → right_shoulder
        (0, 2),  # neck → pelvis
        (3, 4),  # left_shoulder → left_elbow
        (4, 5),  # left_elbow → left_wrist
        (9, 10),  # right_shoulder → right_elbow
        (10, 11),  # right_elbow → right_wrist
        (2, 6),  # pelvis → left_hip
        (6, 7),  # left_hip → left_knee
        (7, 8),  # left_knee → left_ankle
        (2, 12),  # pelvis → right_hip
        (12, 13),  # right_hip → right_knee
        (13, 14),  # right_knee → right_ankle
        (1, 15),  # nose → left_eye
        (15, 16),  # left_eye → left_ear
        (1, 17),  # nose → right_eye
        (17, 18),  # right_eye → right_ear
    ],
)
