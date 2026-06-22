from copy import deepcopy
from dataclasses import dataclass
import os
from os.path import join, realpath
from typing import List, Dict

from constants import SCENE_DIR, REF_DIR

@dataclass
class SceneConfig():
    """Holds configuration options related to each scene."""
    name: str
    fname: str
    param_keys: List[str]
    normal_scene_vars: Dict
    sensors: List[int]
    start_from_value: Dict

    max_depth: int = 64
    # Directory name where to find the reference images.
    # Useful if multiple configurations share the same
    # set of ref images.
    references: str = None
    ref_spp: int = 8192
    ref_integrator: str = 'volpathsimple'
    ref_fname: str = None
    ref_scene_vars: Dict = None
    preview_sensors: List[int] = None
    preview_train_sensors: List[int] = None  # sensor ∈ training set (comparable to paper wandb)
    preview_test_sensors: List[int] = None   # sensor ∉ training set (held-out, clean eval)

    # Upper bound on the density, this prevents very large render times.
    # Its value should be chosen based on the scene scale.
    max_density: float = 250
    # Determines the resolution of the majorant supergrid.
    # Will be adjusted at runtime if upsampling is enabled.
    # The supergrid can be disabled by setting the factor to 0.
    majorant_resolution_factor: int = 8

    # Per-parameter factors to apply to the learning rate
    param_lr_factors: Dict = None

    envmap_rot_group: List[int] = None

    load_ckpt_dir: str = None
    load_ckpt_iter: int = None

    # medium_scale used in the OPT XML; must match the XML default so the
    # true density equals grid_tensor × medium_scale.
    medium_scale: float = 1.0
    # medium_scale for the REF XML.  Only set when REF XML uses a different
    # default than OPT (e.g. dragon_media2: OPT=30, REF=1; teapot: OPT=20, REF=0.05).
    # When None, falls back to self.medium_scale for injection.
    ref_medium_scale: float = None

    def __post_init__(self):
        super().__init__()
        self.fname = realpath(join(SCENE_DIR, self.fname))
        if not os.path.isfile(self.fname):
            raise ValueError(f'Scene file not found: {self.fname}')
        if self.ref_fname:
            self.ref_fname = realpath(join(SCENE_DIR, self.ref_fname))
            if not os.path.isfile(self.ref_fname):
                raise ValueError(f'Reference scene file not found: {self.ref_fname}')

        # Ensure medium_scale is always passed to the XML (overrides XML default).
        self.normal_scene_vars['medium_scale'] = self.medium_scale

        if self.ref_scene_vars is None:
            self.ref_scene_vars = deepcopy(self.normal_scene_vars)

        # Also inject into ref_scene_vars (ref may have different medium_scale from OPT).
        ref_ms = self.ref_medium_scale if self.ref_medium_scale is not None else self.medium_scale
        self.ref_scene_vars['medium_scale'] = ref_ms

        for k in self.param_keys:
            if k not in self.start_from_value:
                raise ValueError(f'Parameter "{k}" will be optimized but was not given an initial value in `start_from_value`')

        if self.references is None:
            self.references = join(REF_DIR, self.name)
        elif not os.path.isdir(self.references):
            self.references = join(REF_DIR, self.references)

        if not self.preview_train_sensors:
            self.preview_train_sensors = [self.sensors[0]]
        if not self.preview_test_sensors:
            self.preview_test_sensors = []
        # preview_sensors = ordered union of train + test (deduped), used for rendering
        seen = set()
        combined = []
        for s in self.preview_train_sensors + self.preview_test_sensors:
            if s not in seen:
                seen.add(s)
                combined.append(s)
        self.preview_sensors = combined

        if not self.param_lr_factors:
            self.param_lr_factors = {}
            for k in self.param_keys:
                if '.albedo.' in k:
                    self.param_lr_factors[k] = 2.0


_SCENE_CONFIGS = {}
_SCENE_CONFIG_KWARGS = {}
def add_scene_config(name, **kwargs):
    assert name not in _SCENE_CONFIGS, f'Duplicate scene config name: {name}'
    fname = realpath(join(SCENE_DIR, kwargs['fname']))
    if not os.path.isfile(fname):
        print(f'[Warning] Skipping scene config "{name}": file not found: {fname}')
        _SCENE_CONFIG_KWARGS[name] = deepcopy(kwargs)
        return
    _SCENE_CONFIGS[name] = SceneConfig(name, **kwargs)
    _SCENE_CONFIG_KWARGS[name] = deepcopy(kwargs)

def add_scene_config_variant(name, base, **kwargs):
    assert name not in _SCENE_CONFIGS, f'Duplicate scene config name: {name}'
    all_kwargs = deepcopy(_SCENE_CONFIG_KWARGS[base])
    all_kwargs.update(deepcopy(kwargs))
    fname = realpath(join(SCENE_DIR, all_kwargs['fname']))
    if not os.path.isfile(fname):
        print(f'[Warning] Skipping scene config variant "{name}": file not found: {fname}')
        _SCENE_CONFIG_KWARGS[name] = all_kwargs
        return
    _SCENE_CONFIGS[name] = SceneConfig(name, **all_kwargs)
    _SCENE_CONFIG_KWARGS[name] = all_kwargs


def get_scene_config(name):
    if isinstance(name, SceneConfig):
        return deepcopy(name)
    return deepcopy(_SCENE_CONFIGS[name])



# ----------

add_scene_config(
    'dust-devil',
    fname='dust-devil/dust-devil.xml',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data', 'medium1.emission.data'],
    medium_scale=100.0,
    normal_scene_vars={
        'resx': 620,
        'resy': 720,
        # 256^3 to match the final resolution produced by the
        # dust-devil-l1-1e-2-formal-local-single-gpu-256-0.3lrdown wandb runs.
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/kloofendal_38d_partly_cloudy_4k.exr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 620,
        'resy': 720,
        'medium_filename': 'volumes/embergen_dust_devil_tornado_a_50-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'emission_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/kloofendal_38d_partly_cloudy_4k.exr',
        'majorant_resolution_factor': 8,
    },
    sensors=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04 / 100,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1 / 100,
    },
    preview_train_sensors=[1],  # paper eval sensor (∈ training)
    preview_test_sensors=[0],   # held-out test view (∉ training)
)

add_scene_config(
    'jellyfish',
    fname='jellyfish/jellyfish.xml',
    # Reference is rendered through the OPT XML against the GT volumes
    # (true density + albedo) that live in the scene dir, at medium_scale 1.0.
    ref_integrator='volpathsimple',
    ref_medium_scale=1.0,
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data'],
    medium_scale=20.0,
    normal_scene_vars={
        'resx': 720,
        'resy': 720,
        'medium_filename': 'volumes/sigma_t-constant-sand-320-320-320.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'emission_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 720,
        'resy': 720,
        'medium_filename': 'jellyfish-true-density.vol',
        'albedo_filename': 'jellyfish-albedo.vol',
        'emission_filename': 'jellyfish-albedo.vol',
        'majorant_resolution_factor': 8,
    },

    sensors=[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04 / 20,
        'medium1.albedo.data': 0.6,
    },
    preview_train_sensors=[0],  # paper eval sensor (∈ training)
    preview_test_sensors=[3],   # held-out test view (∉ training)
)




add_scene_config(
    'dragon_media2',
    fname='dragon_media2/dragon_media2.xml',
    # Reference uses the legacy mesh-bounded scene (scene.xml), which hardcodes
    # the GT volumes (volume/sigma_sym.vol, volume/albedo_sym.vol), dragon.obj
    # and high_sun.exr relative to the scene dir. NOTE: this is an APPROXIMATE
    # reproduction of mi_ref/dragon_media2 (the authoritative render was done on
    # a different machine); the dragon shape/pose/exposure match but expect
    # minor silhouette differences.
    ref_fname='dragon_media2/scene.xml',
    ref_integrator='volpathsimple',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data'],
    medium_scale=30.0,       # OPT XML default; true sigma_t = grid × 30
    ref_medium_scale=1.0,    # legacy scene.xml renders the GT at scale 1.0
    normal_scene_vars={
        'resx': 768,
        'resy': 576,
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 768,
        'resy': 576,
    },

    sensors=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04/30,
        'medium1.albedo.data': 0.6,
        # 'medium1.emission.data': 0.1 / 100,
    },
    preview_train_sensors=[62],  # paper eval sensor (∈ training)
    preview_test_sensors=[0],    # held-out test view (∉ training)
)


add_scene_config(
    'teapot',
    fname='teapot/teapot.xml',
    # Reference is rendered through the OPT XML against the GT volumes
    # (true density + albedo) in the scene dir, at medium_scale 1.0.
    ref_integrator='volpathsimple',
    medium_scale=20.0,       # OPT XML default
    ref_medium_scale=1.0,
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data'],
    normal_scene_vars={
        'resx': 683,
        'resy': 512,
        'medium_filename': 'volumes/sigma_t-constant-sand-320-320-320.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 683,
        'resy': 512,
        'medium_filename': 'teapot-true-density.vol',
        'albedo_filename': 'teapot-albedo.vol',
        'majorant_resolution_factor': 8,
    },

    sensors=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04/20,
        'medium1.albedo.data': 0.6,
    },
    preview_train_sensors=[4],  # paper eval sensor (∈ training)
    preview_test_sensors=[0],   # held-out test view (∉ training)
)



add_scene_config(
    'scarf',
    fname='scarf/scarf.xml',
    ref_fname='scarf/scarf-ref.xml',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data'],
    medium_scale=80.0,
    normal_scene_vars={
        'resx': 768,
        'resy': 576,
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        # 'envmap_filename': 'textures/underwater.exr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 768,
        'resy': 576,
        # 'envmap_filename': 'textures/underwater.exr',
        'majorant_resolution_factor': 8,
    },

    sensors=[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04/80,
        'medium1.albedo.data': 0.6,
    },
    preview_train_sensors=[0],  # paper eval sensor (∈ training)
    preview_test_sensors=[2],   # held-out test view (∉ training)
)

add_scene_config(
    'bunny-cloud',
    fname='bunny-cloud/bunny-cloud.xml',
    ref_fname='bunny-cloud/bunny-cloud-ref.xml',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data'],
    medium_scale=60.0,
    normal_scene_vars={
        'resx': 768,
        'resy': 576,
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        # 'envmap_filename': 'textures/underwater.exr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 768,
        'resy': 576,
        # 'envmap_filename': 'textures/underwater.exr',
        'majorant_resolution_factor': 8,
    },

    sensors=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04/60,
        'medium1.albedo.data': 0.6,
    },
    preview_train_sensors=[1],  # paper eval sensor (∈ training)
    preview_test_sensors=[0],   # held-out test view (∉ training)
)


# ----------
add_scene_config(
    'astronaut-rotated',
    fname='astronaut-rotated/astronaut-rotated.xml',
    ref_fname='astronaut-rotated/astronaut-rotated-ref.xml',
    ref_integrator='path',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data'],
    medium_scale=100.0,
    normal_scene_vars={
        'resx': 720,
        'resy': 1080,
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/kloofendal_38d_partly_cloudy_4k.exr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 720,
        'resy': 1080,
        'envmap_filename': 'textures/kloofendal_38d_partly_cloudy_4k.exr',
    },
    sensors=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04 / 100,
        'medium1.albedo.data': 0.6,
    },
    preview_train_sensors=[3],  # train preview sensor (∈ training)
    preview_test_sensors=[1],   # held-out test view — sensor 1 has clean convergence curve
)
# ----------

add_scene_config(
    'rover',
    fname='rover/rover.xml',
    ref_fname='rover/rover-ref.xml',
    ref_integrator='path',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data', 'medium1.emission.data'],
    medium_scale=5.0,
    normal_scene_vars={
        'resx': 860,
        'resy': 720,
        # 320^3 to match the final resolution produced by the
        # rover-l1-1e-2-formal-local-single-gpu wandb runs.
        'medium_filename': 'volumes/sigma_t-constant-sand-320-320-320.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'emission_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/gamrig_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 860,
        'resy': 720,
        'envmap_filename': 'textures/gamrig_2k.hdr',
    },
    sensors=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    # envmap_rot_group=[0,1,2,3,4,5],
    start_from_value={
        'medium1.sigma_t.data': 0.04/5.0,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1,
    },
    preview_train_sensors=[1],  # paper eval sensor (∈ training)
    preview_test_sensors=[0],   # held-out test view (∉ training)
)


