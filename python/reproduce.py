"""
Reproduces results from the article:
    Merlin Nimier-David, Thomas Müller, Alexander Keller, and Wenzel Jakob. 2022.
    Unbiased Inverse Volume Rendering with Differential Trackers.
    In Transactions on Graphics (Proceedings of SIGGRAPH) 41(4).
"""

from copy import deepcopy
import os
from os.path import join

import mitsuba as mi

from constants import OUTPUT_DIR
from opt_config import get_int_config, OptimizationConfig, Schedule
from optimize import run_optimization
from scene_config import get_scene_config
import losses


def reproduce_optimization_experiments(configs, overwrite=False, use_log=False, use_rot_envmap=False):
    for cname, entries in configs.items():
        exp_output_dir = join(OUTPUT_DIR, cname)
        scene_config = get_scene_config(entries['scene'])

        for int_name, opt_overrides in entries['integrators'].items():
            int_config = get_int_config(int_name)
            opt_config = deepcopy(entries['opt'])
            if opt_overrides:
                opt_config.update(opt_overrides)
            opt_config = OptimizationConfig(name=cname, **opt_config)

            output_dir = join(exp_output_dir, int_name)
            os.makedirs(output_dir, exist_ok=True)

            result_fname = join(output_dir, 'params', 'final-medium1_sigma_t.vol')
            if overwrite or not os.path.isfile(result_fname):
                run_optimization(output_dir, opt_config, scene_config, int_config, use_log=use_log, use_rot_envmap=use_rot_envmap)


#
# Four "standard" integrators used by every formal scene experiment below.
# Within each formal experiment all integrators share the same hyper-parameters.
#
FORMAL_INTEGRATORS = [
    'volpathsimple-drt-mis-n4',         # baseline quadratic
    'volpathsimple-drt-mis-linear',     # baseline linear
    'volpathfm-drt-sd-n4',              # ours quadratic (pure matched, no MIS-mix)
    'volpathfm-linear-drt-sd-n4',       # ours linear   (pure matched, no MIS-mix)
]


def _formal_experiment(scene, lr, upsample, upsample_lr_factor=0.5):
    """Build a config entry matching the `formal-local-single-gpu` wandb runs.

    Hyper-parameters common to every formal run are defined here; only the
    per-scene knobs (scene, lr, upsample, upsample_lr_factor) vary.  Every
    standard integrator shares the same opt config.
    """
    opt = {
        'n_iter': 8000,
        'preview_stride': 250,
        'spp': 16,
        'primal_spp_factor': 64,
        'lr': lr,
        'lr_schedule': Schedule.Constant,
        'batch_size': 32768,
        'importance_iter_stride': None,
        'importance_iter': 400,
        'importance_grid_resolution': [32, 32, 32],
        'render_initial': True,
        'render_final': True,
        'preview_spp': 1024,
        'checkpoint_stride': 2000,
        'loss': losses.l1,
        'upsample': upsample,
        'upsample_lr_factor': upsample_lr_factor,
        'save_opt_state': False,
    }
    return {
        'scene': scene,
        'opt': opt,
        'integrators': {name: None for name in FORMAL_INTEGRATORS},
    }


# Optimization config name → {scene, opt, integrators}.
# These names match the wandb run names and the `picked_ckpt2/<name>/` folders
# on disk.  Each entry reproduces one column of the teaser table (one scene ×
# four standard integrators).
FORMAL_CONFIGS = {
    # scene final sigma_t res: 256^3, init 64^3
    'bunny-cloud-l1-6e-3-formal-local-single-gpu': _formal_experiment(
        scene='bunny-cloud', lr=6e-3, upsample=[0.01, 0.02]),
    # final 256^3, init 64^3
    'astronaut-rotated-l1-5e-3-formal-local-single-gpu': _formal_experiment(
        scene='astronaut-rotated', lr=5e-3, upsample=[0.01, 0.02]),
    # final 320^3, init 80^3
    'jellyfish-l1-6e-3-formal-local-single-gpu': _formal_experiment(
        scene='jellyfish', lr=6e-3, upsample=[0.005, 0.01]),
    # final 320^3, init 40^3 (3 upsamples)
    'teapot-l1-3e-3-formal-local-single-gpu-3': _formal_experiment(
        scene='teapot', lr=3e-3, upsample=[0.02, 0.04, 0.08]),
    # final 256^3, init 32^3 (3 upsamples)
    'scarf-l1-3e-3-formal-local-single-gpu': _formal_experiment(
        scene='scarf', lr=3e-3, upsample=[0.02, 0.04, 0.08]),
    # final 320^3, init 80^3
    'rover-l1-1e-2-formal-local-single-gpu': _formal_experiment(
        scene='rover', lr=1e-2, upsample=[0.02, 0.04]),
    # final 256^3, init 64^3
    'dragon_media2-l1-1e-2-formal': _formal_experiment(
        scene='dragon_media2', lr=1e-2, upsample=[0.005, 0.01]),
    # final 256^3, init 64^3 — per-upsample lr factor 0.3 (the "-0.3lrdown" suffix).
    'dust-devil-l1-1e-2-formal-local-single-gpu-256-0.3lrdown': _formal_experiment(
        scene='dust-devil', lr=1e-2, upsample=[0.01, 0.02],
        upsample_lr_factor=0.3),
}


def main(opt_config_name=None, integrators=None, use_log=False, use_rot_envmap=False):
    base_opt_config = {
        'n_iter': 8000,
        'preview_stride': 250,
        'spp': 16,
        'primal_spp_factor': 64,
        'lr': 5e-2,
        'lr_schedule': Schedule.Constant,
        'batch_size': 32768,
        'importance_iter_stride': None,
        'importance_iter': 400,
        'importance_grid_resolution': [32, 32, 32],
        'render_initial': True,
        'render_final': True,
        'preview_spp': 1024,
        'checkpoint_stride': 2000,
        'loss': losses.l1,
        'upsample': [0.01, 0.02],
    }

    # Structure of this dictionary:
    # Optimization config name: {
    #    'scene': scene config name,
    #    'opt': optimization configuration,
    #    'integrators': {
    #        Integrator name: dict of per-integrator optimization config
    #                         overrides (or None if no overrides are needed),
    #    }
    # }
    configs = {
        # Legacy single-GPU CPU-6430 sweep (kept for backward compatibility).
        'bunny-cloud-l1-6e-3-formal-single-cpu-6430': {
            'scene': 'bunny-cloud',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'volpathsimple-drt-mis-n4':        {'lr': 6e-3},
                'volpathfm-drt-sd-n4':             {'lr': 6e-3},
                'volpathsimple-drt-mis-linear':    {'lr': 6e-3},
                'volpathfm-linear-drt-sd-n4':      {'lr': 6e-3},
            },
        },
    }
    # The 8 formal experiments that reproduce the wandb runs mapped in
    # `ckpt_run_mapping.md`.
    configs.update(deepcopy(FORMAL_CONFIGS))

    # Filter to the selected config and integrators, if specified
    if opt_config_name is not None:
        assert opt_config_name in configs, \
               f'Configuration name "{opt_config_name}" not found. Available configs: {list(configs.keys())}.'
        configs = { opt_config_name: configs[opt_config_name] }

    if integrators is not None:
        configs = deepcopy(configs)
        found = False
        for _, c in configs.items():
            c['integrators'] = { k: i for k, i in c['integrators'].items()
                                 if k in integrators }
            found = found or (len(c) > 0)

        assert found, f'No configuration using integrator name "{opt_config_name}" found.'

    reproduce_optimization_experiments(configs, overwrite=False, use_log=use_log, use_rot_envmap=use_rot_envmap)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('reproduce.py')
    parser.add_argument('--config', type=str, dest='opt_config_name', default=None,
                        help='Optimization configuration name to run. If not specified, all configurations will run.')
    parser.add_argument('--integrator', type=str, action='append', dest='integrators', default=None,
                        help='Subset of integrators (methods) to use. If not specified, all available methods will run.')
    parser.add_argument('--log', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--use-rot-envmap', action='store_true', help='Use rotated environment map for the dust devil scene')
    parser.add_argument('--list', action='store_true', help='List all available optimization config names and exit.')
    args = parser.parse_args()

    if args.list:
        print('Available optimization configs:')
        print('  bunny-cloud-l1-6e-3-formal-single-cpu-6430  (legacy)')
        for name in FORMAL_CONFIGS:
            print(f'  {name}')
        raise SystemExit(0)

    mi.set_variant('cuda_ad_rgb')
    main(opt_config_name=args.opt_config_name, integrators=args.integrators, use_log=args.log, use_rot_envmap=args.use_rot_envmap)
