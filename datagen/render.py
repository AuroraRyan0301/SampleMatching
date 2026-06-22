import os
from os.path import realpath, join, dirname
import sys

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')  # or 'cuda_ad_rgb', etc.

from pathlib import Path
import numpy as np

def cube_test_scene(resx=128, resy=128, spp=16, pixel_format='rgb', sample_emitters=True,
                    density_scale=1.0, use_fast_path=False):
    T = mi.ScalarTransform4f

    grids = [np.full((3, 3, 3, k), 1.0, dtype=np.float32) for k in (1, 3)]
    # Add some basic spatial variations
    grids[0] *= 0.5
    grids[0][0, 0, 0, :] = 0.1
    grids[0][0, -1, 0, :] = 2.0
    grids[0][0, 0, -1, :] = 0.2
    grids[1][..., 0] = 0.3
    grids[1][..., 1] = 0.5
    grids[1][..., 2] = 0.9
    for i in range(grids[1].shape[0]):
        grids[1][i, :, :, 0] *= np.square((i+1) / grids[1].shape[0])
        grids[1][i, :, :, 1] *= 1 - (i+1) / grids[1].shape[0]
        grids[1][:, i, :, 1] *= np.square((i+1) / grids[1].shape[0])
    # Albedo grid
    grids.append(np.clip(grids[1], 0, 1))

    grids = [mi.VolumeGrid(g) for g in grids]
    to_world = T.translate([-0.5, -0.5, -0.5]).scale([2, 2, 2])

    return {
        'type': 'scene',
        'use_bbox_fast_path': True,
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov': 30,
            'to_world': T.look_at(
                origin=[4.0, 4.0, 4.0],
                target=[0, -0.15, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': spp,
            },
            'film': {
                'type': 'hdrfilm',
                'width' : resx,
                'height': resy,
                'rfilter': {
                    'type': 'box',
                },
                'pixel_format': pixel_format,
            }
        },
        # Mostly just to avoid the warning
        'integrator': {
            'type': 'volpath',
        },
        # -------------------- Light --------------------
        'light': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': [1.0, 0.8, 0.2]},
            # TODO: switch to an envmap
        },
        # -------------------- Media --------------------
        'medium1': {
            'type': 'heterogeneous',
            'sample_emitters': sample_emitters,
            'has_spectral_extinction': False,
            'scale': density_scale,
            # 'albedo': {
            #     'type': 'constvolume',
            #     'value': {'type': 'rgb', 'value': [0.8, 0.9, 0.7]},
            # },
            'sigma_t': {
                'type': 'gridvolume',
                'grid': grids[0],
                'to_world': to_world,
            },
            'emission': {
                'type': 'gridvolume',
                'grid': grids[1],
                'to_world': to_world,
            },
            'albedo': {
                'type': 'gridvolume',
                'grid': grids[2],
                'to_world': to_world,
            },
        },
        # -------------------- Shapes --------------------
        'cube': {
            # Cube covers [0, 0, 0] to [1, 1, 1] by default
            'type': 'obj',
            'filename': 'cube_unit.obj',
            'bsdf': { 'type': 'null', },
            'interior': {
                'type': 'ref',
                'id':  'medium1'
            },
            'to_world': to_world,
        },
    }


def render_all_sensors(scene_file, output_dir="renders"):
    # scene = mi.load_file(scene_file)
    scene = mi.load_dict(cube_test_scene())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mitsuba 3: sensors are accessible via scene.sensors()
    sensors = scene.sensors()

    print(f"Found {len(sensors)} sensors")

    for i, sensor in enumerate(sensors):
        # Optional: get ID from properties
        sensor_id = f"sensor_{i}"

        print(f"Rendering sensor {i} (id={sensor_id})")

        image = mi.render(scene, sensor=sensor, spp=1024)

        out_path = output_dir / f"{sensor_id}.exr"
        mi.Bitmap(image).write(str(out_path))

    print("Done.")

if __name__ == "__main__":
    render_all_sensors("scene.xml", output_dir="views")