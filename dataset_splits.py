"""List of subsets."""

DATASETS = {
    'pouring': {'train': 70, 'val': 14, 'test': 32},
    'baseball_pitch': {'train': 103, 'val': 63, 'max_len':243},
    'baseball_swing': {'train': 113, 'val': 57,'max_len':95},
    'bench_press': {'train': 113, 'val': 57, 'max_len':218},
    'bowling': {'train': 134, 'val': 84,'max_len':564},
    'clean_and_jerk': {'train': 40, 'val': 42,'max_len':663},
    'golf_swing': {'train': 87, 'val': 77,'max_len':95},
    'jumping_jacks': {'train': 56, 'val': 56,'max_len':42},
    'pushups': {'train': 102, 'val': 105,'max_len':189},
    'pullups': {'train': 98, 'val': 100,'max_len':301},
    'situp': {'train': 50, 'val': 50,'max_len':242},
    'squats': {'train': 114, 'val': 115,'max_len':178},
    'tennis_forehand': {'train': 79, 'val': 74,'max_len':95},
    'tennis_serve': {'train': 115, 'val': 68,'max_len':100},
    'kallax_shelf_drawer': {'train': 61, 'val': 29,'max_len':4078},
    'pouring_milk': {'train': 27, 'val': 11,'max_len':865},
}


DATASET_TO_NUM_CLASSES = {
    'pouring': 5,
    'baseball_pitch': 4,
    'baseball_swing': 3,
    'bench_press': 2,
    'bowling': 3,
    'clean_and_jerk': 6,
    'golf_swing': 3,
    'jumping_jacks': 4,
    'pushups': 2,
    'pullups': 2,
    'situp': 2,
    'squats': 4,
    'tennis_forehand': 3,
    'tennis_serve': 4,
    'Kallax_Shelf_Drawer': 17,
    'pouring_milk': 10,
}
