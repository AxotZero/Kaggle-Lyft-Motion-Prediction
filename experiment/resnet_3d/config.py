# --- Lyft configs ---
DEBUG = False
GPU = True
cfg = {
    'format_version': 4,
    'data_path': "/home/axot/lyft/data",
    'model_params': {
        'model_depth': 18,
        'history_num_frames': 10,
        'future_num_frames': 50,
        'lr': 1e-3,
        # 'weight_path': "/home/axot/lyft/experiment/resnet_3d/epoch01_iter69999.pth",
        "weight_path": "/home/axot/lyft/experiment/resnet_3d/pretrain_model/112_v5.pth",
        # "weight_path": "",
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_step_size': 1,
        'future_delta_time': 0.1,
    },
    'raster_params': {
        'raster_size': [112, 112],
        'pixel_size': [1.0, 1.0],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        # 'filter_agents_threshold': 0.5,
        'filter_agents_threshold': 0.7,
        'disable_traffic_light_faces': False
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 2
    },
    'val_data_loader':{
        'key': 'scenes/validate.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 4
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'epoch': 1 if DEBUG else 10,
        'max_num_steps': 4 if DEBUG else 57_001,
        'checkpoint_steps': 2 if DEBUG else 5_000,
        'validation_steps': 1 if DEBUG else 10_000,
    }
}