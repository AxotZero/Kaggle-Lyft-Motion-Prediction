# --- Lyft configs ---
DEBUG = False
GPU = True
cfg = {
    'format_version': 4,
    'data_path': "/home/axot/lyft/data",
    'model_params': {
        'model_architecture': 'efficientnet-b1',
        'history_num_frames': 30,
        'future_num_frames': 50,
        'lr': 1e-3,
        'weight_path': "/home/axot/lyft/experiment/three_eff_f30_e5_s20000/pretrained_models/pretrained2_epoch01_iter49999.pth",
        # 'ignore_weight_keys': [
        #     '_fc.weight',
        #     '_fc.bias'
        # ],
        'history_step_size': 1,
        'history_delta_time': 0.2,
        'future_step_size': 1,
        'future_delta_time': 0.1,
    },
    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
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
        'batch_size': 11,
        'shuffle': True,
        'num_workers': 4
    },
    'val_data_loader':{
        'key': 'scenes/validate.zarr',
        'batch_size': 11,
        'shuffle': True,
        'num_workers': 4
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'epoch': 2 if DEBUG else 2,
        'max_num_steps': 2 if DEBUG else 75_000,
        'checkpoint_steps': 1 if DEBUG else 25_000,
        # 'val_num_steps': 5 if DEBUG else 3_000,
    }
}