# --- Lyft configs ---
DEBUG = False
GPU = True
cfg = {
    'format_version': 4,
    'data_path': "/home/axot/lyft/data",
    'model_params': {
        'model_depth': 18,
        'history_num_frames': 20,
        'future_num_frames': 50,
        'lr': 1e-3,
        'weight_path': "/home/axot/lyft/experiment/three_resnet18_3d_frame20/save_models/interrupt_epoch00_iter57739.pth",
        "ignore_weight_keys":[
            "conv1.weight",
            "fc.weight",
            "fc.bias"
        ],
        'history_step_size': 1,
        'history_delta_time': 0.3,
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
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces': False
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 8
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'epoch': 1 if DEBUG else 2,
        'max_num_steps': 4 if DEBUG else 140_000,
        'checkpoint_steps': 2 if DEBUG else 26_000,
    }
}