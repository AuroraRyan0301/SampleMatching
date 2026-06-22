"""
Relighting Pipeline A — relight 8 scenes with a new envmap (hansaplatz).

For each (scene, integrator) run:
  * Load scene XML pointing at the OPT final-*.vol from `output/`
  * Override envmap (filename + scale + per-scene rotation + res + fov)
  * Render one preview sensor (held-out sensor 0 usually) → OPT render
Also for each scene, render the GT reference using the scene's GT volumes.

Adapted from the original `relighting_scene.py` (paper internal) with these
changes:
  * No teaser / dragon_media branches (Pipeline A only)
  * ckpt source: our `output/<config_name>/<integrator>/params/final-*.vol`
  * envmap default: repo-local hansaplatz_4k.exr
  * spp is configurable; default low (64) for preview
"""

import os
import argparse
from os.path import join

import drjit as dr
import mitsuba as mi

from opt_config import get_int_config
from scene_config import get_scene_config
from constants import OUTPUT_DIR, DATA_DIR
# NOTE: `import integrators` is deferred to main() because it needs
# mi.set_variant(...) to be called first.

# Default relighting envmap (e.g. PolyHaven "hansaplatz 4k"). Place it in the
# data dir, or pass --envmap to override.
DEFAULT_ENVMAP = join(DATA_DIR, 'envmaps', 'hansaplatz_4k.exr')
DEFAULT_ENVMAP_SCALE = 2.0

# Scene → config_name mapping (matches our output/ directory)
SCENE_TO_CONFIG = {
    'bunny-cloud':       'bunny-cloud-l1-6e-3-formal-local-single-gpu',
    'astronaut-rotated': 'astronaut-rotated-l1-5e-3-formal-local-single-gpu',
    'jellyfish':         'jellyfish-l1-6e-3-formal-local-single-gpu',
    'teapot':            'teapot-l1-3e-3-formal-local-single-gpu-3',
    'scarf':             'scarf-l1-3e-3-formal-local-single-gpu',
    'rover':             'rover-l1-1e-2-formal-local-single-gpu',
    'dust-devil':        'dust-devil-l1-1e-2-formal-local-single-gpu-256-0.3lrdown',
    'dragon_media2':     'dragon_media2-l1-1e-2-formal',
}

# Per-scene envmap rotation overrides (from the original relighting_scene.py)
ENVMAP_ROT = {
    'teapot':            (0,    0, 180),
    'rover':             (0,    0, 180),
    'dragon_media2':     (90,   0,   0),
    'scarf':             (-90,  0,   0),
    'bunny-cloud':       (-90,  0,   0),
    'astronaut-rotated': (0,   90,   0),
    'jellyfish':         (-90,  0,   0),
    'dust-devil':        (0,  120,   0),  # (x,z,y) in original; z swapped below
}
# (x, z, y) tuples above; convert to explicit x/y/z dict per scene
ENVMAP_ROT_XYZ = {
    'teapot':            dict(x=0,  y=180, z=0),
    'rover':             dict(x=0,  y=180, z=0),
    'dragon_media2':     dict(x=90, y=0,   z=0),
    'scarf':             dict(x=-90,y=0,   z=0),
    'bunny-cloud':       dict(x=-90,y=0,   z=0),
    'astronaut-rotated': dict(x=0,  y=0,   z=90),
    'jellyfish':         dict(x=-90,y=0,   z=0),
    'dust-devil':        dict(x=0,  y=0,   z=120),
}

# Per-scene FOV overrides for the relighting view
FOV_OVERRIDE = {
    'astronaut-rotated': 50.0,
    'jellyfish':         50.0,
}

INTEGRATORS = [
    'volpathsimple-drt-mis-n4',
    'volpathsimple-drt-mis-linear',
    'volpathfm-drt-mis-sd-n4',
    'volpathfm-linear-drt-mis-sd-n4',
]


def find_final_vol(params_dir, suffix):
    """Look for final-medium1_<suffix>.vol, fall back to latest timestamped."""
    final = join(params_dir, f'final-medium1_{suffix}.vol')
    if os.path.isfile(final):
        return final
    # Fallback: pick lexicographically last timestamped file
    import glob
    cands = sorted(glob.glob(join(params_dir, f'*medium1_{suffix}.vol')))
    if not cands:
        raise FileNotFoundError(f'No *medium1_{suffix}.vol in {params_dir}')
    return cands[-1]


def pick_preview_sensor(scene_config):
    """Use an id NOT in training sensors as the held-out relight preview view."""
    existing = set(scene_config.sensors)
    # Scan 0..max for the first missing id
    for i in range(max(existing) + 2):
        if i not in existing:
            return i
    return scene_config.sensors[0]  # fallback


def apply_relighting_vars(scene_config, envmap_path, envmap_scale, resx, resy):
    rot = ENVMAP_ROT_XYZ[scene_config.name]
    fov = FOV_OVERRIDE.get(scene_config.name, None)

    for vars_dict in (scene_config.normal_scene_vars, scene_config.ref_scene_vars):
        vars_dict['envmap_filename']   = envmap_path
        vars_dict['envmap_scale']      = envmap_scale
        vars_dict['envmap_x_rotation'] = rot['x']
        vars_dict['envmap_y_rotation'] = rot['y']
        vars_dict['envmap_z_rotation'] = rot['z']
        vars_dict['resx'] = resx
        vars_dict['resy'] = resy
        if fov is not None:
            vars_dict['fov'] = fov


def load_scene(scene_config, reference=False, **kwargs):
    scene_vars = scene_config.ref_scene_vars if reference else scene_config.normal_scene_vars
    fname = scene_config.ref_fname if reference and scene_config.ref_fname else scene_config.fname
    return mi.load_file(fname, **scene_vars, **kwargs)


def render_view(scene, sensor_idx, integrator, spp, out_path, seed=1234):
    sensor = scene.sensors()[sensor_idx]
    img = mi.render(scene, sensor=sensor, integrator=integrator, spp=spp, seed=seed)
    dr.eval(img)
    mi.Bitmap(img).write(out_path)


def relight_one(scene_name, integ_name, out_root, opt_output_dir,
                envmap_path, envmap_scale, spp, resx, resy, seed=1234,
                skip_gt=False):
    scene_config = get_scene_config(scene_name)
    config_name = SCENE_TO_CONFIG[scene_name]

    # Point the scene XML's medium/albedo files at OPT final volumes.
    params_dir = join(opt_output_dir, config_name, integ_name, 'params')
    sigma_t_vol = find_final_vol(params_dir, 'sigma_t')
    albedo_vol  = find_final_vol(params_dir, 'albedo')

    scene_config.normal_scene_vars['medium_filename'] = sigma_t_vol
    scene_config.normal_scene_vars['albedo_filename'] = albedo_vol

    apply_relighting_vars(scene_config, envmap_path, envmap_scale, resx, resy)

    # Preview sensor: first id NOT in training
    preview = pick_preview_sensor(scene_config)
    if preview not in scene_config.sensors:
        scene_config.sensors = sorted(set(scene_config.sensors) | {preview})

    # OPT render
    scene = load_scene(scene_config, reference=False)
    int_config = get_int_config(integ_name)
    integrator = int_config.create(max_depth=scene_config.max_depth)
    out_dir = join(out_root, f'{scene_name}-render', integ_name)
    os.makedirs(out_dir, exist_ok=True)
    render_view(scene, preview, integrator, spp,
                join(out_dir, f'opt_{preview:06d}.exr'), seed=seed)
    # With no background (hide_emitters)
    integrator.hide_emitters = True
    render_view(scene, preview, integrator, spp,
                join(out_dir, f'opt_{preview:06d}_nobg.exr'), seed=seed)

    if skip_gt:
        return preview

    # ---- GT render ----
    # Prefer OPT-coord GT cache (gt_true_density/) because it matches OPT's
    # coordinate system and is already scaled to true density. Fall back to
    # REF XML for scenes whose ref uses a mesh (astronaut/jellyfish/rover)
    # or where no gt_true_density cache exists.
    use_gt_cache = bool(scene_config.gt_sigma_t_path and
                        os.path.isfile(scene_config.gt_sigma_t_path))

    if use_gt_cache:
        # Swap OPT XML's medium to the GT true-density vol; medium_scale=1 since
        # the cache already contains true σ_t.
        scene_config.normal_scene_vars['medium_filename'] = scene_config.gt_sigma_t_path
        scene_config.normal_scene_vars['medium_scale']   = 1.0
        if scene_config.gt_albedo_path and os.path.isfile(scene_config.gt_albedo_path):
            scene_config.normal_scene_vars['albedo_filename'] = scene_config.gt_albedo_path
        scene = load_scene(scene_config, reference=False)
        int_cfg = get_int_config('volpathsimple-drt-mis-n4')
        gt_int = int_cfg.create(max_depth=scene_config.max_depth)
    else:
        scene = load_scene(scene_config, reference=True)
        gt_int = mi.load_dict({
            'type': scene_config.ref_integrator,
            'max_depth': scene_config.max_depth,
        })

    gt_dir = join(out_root, f'{scene_name}-render', 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    render_view(scene, preview, gt_int, spp,
                join(gt_dir, f'gt_{preview:06d}.exr'), seed=seed)

    # no-bg variant
    if use_gt_cache:
        gt_int.hide_emitters = True
        nobg_int = gt_int
    else:
        nobg_int = mi.load_dict({
            'type': scene_config.ref_integrator,
            'max_depth': scene_config.max_depth,
            'hide_emitters': True,
        })
    gt_nobg_dir = join(out_root, f'{scene_name}-render', 'gt_nobg')
    os.makedirs(gt_nobg_dir, exist_ok=True)
    render_view(scene, preview, nobg_int, spp,
                join(gt_nobg_dir, f'gt_{preview:06d}_nobg.exr'), seed=seed)
    return preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True,
                    help=f'Scene name, one of {list(SCENE_TO_CONFIG)}')
    ap.add_argument('--integrator', default=None,
                    help='Single integrator to render (default: all 4)')
    ap.add_argument('--out-dir', default='/scratch/bgcf/ling23/posttracking-reproduce/relighting')
    ap.add_argument('--opt-output-dir', default=OUTPUT_DIR,
                    help='Repo output/ root containing <config>/<integ>/params/')
    ap.add_argument('--envmap', default=DEFAULT_ENVMAP)
    ap.add_argument('--envmap-scale', type=float, default=DEFAULT_ENVMAP_SCALE)
    ap.add_argument('--spp', type=int, default=64)
    ap.add_argument('--resx', type=int, default=1920)
    ap.add_argument('--resy', type=int, default=1440)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--skip-gt', action='store_true',
                    help='Skip GT rendering (when already rendered earlier)')
    ap.add_argument('--variant', default='cuda_ad_rgb')
    args = ap.parse_args()

    mi.set_variant(args.variant)
    # Custom integrators require a variant to be set first.
    import integrators  # noqa: F401

    integs = [args.integrator] if args.integrator else INTEGRATORS
    print(f'[relight] scene={args.scene}  integrators={integs}  spp={args.spp}')

    for i, integ in enumerate(integs):
        # GT render only needs to happen once per scene
        skip_gt = args.skip_gt or (i > 0)
        preview = relight_one(
            args.scene, integ, args.out_dir, args.opt_output_dir,
            args.envmap, args.envmap_scale, args.spp, args.resx, args.resy,
            seed=args.seed, skip_gt=skip_gt,
        )
        print(f'  [OK] {integ}  preview={preview}')

    print(f'[done] output: {args.out_dir}/{args.scene}-render/')


if __name__ == '__main__':
    main()
