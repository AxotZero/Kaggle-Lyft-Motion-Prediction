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
        'weight_path': "/home/axot/lyft/experiment/eff_f30_e5_s10000/save_models/pretrained_epoch04_iter09999.pth",
        
        'history_step_size': 1,
        'history_delta_time': 0.3,
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
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4
    },
    'sample_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4
    },
    'train_params': {
        'epoch': 5 if DEBUG else 6,
        'max_num_steps': 10 if DEBUG else 10_000,
        'checkpoint_steps': 5 if DEBUG else 4_000,
    }
}