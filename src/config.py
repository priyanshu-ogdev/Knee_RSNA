import os

# Base configuration
SEED = 2026
CROP_MM = 130.0
IMG_SIZE = 336
GROUP_SIZE = 3  # slices per slot

# Training configuration
EPOCHS = 10
BATCH_SIZE = 8
LR_HEAD = 0.001
LR_BACKBONE = 8e-06
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02

# Augmentation configuration
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1

# Targets
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 
    'Contusion', 'Fracture'
]

# Standardized acquisition slots
SLOTS = [
    ('SAG_FLUID_FS', 'Sagittal', True, True),
    ('COR_FLUID_FS', 'Coronal', True, True),
    ('AX_FLUID_FS', 'Axial', True, True),
    ('SAG_FLUID_NOFS', 'Sagittal', True, False),
    ('COR_T1', 'Coronal', False, False),
    ('SAG_T1', 'Sagittal', False, False)
]
N_SLOTS = len(SLOTS)
