import pickle
from constants import WANDB_EXP_NAME
from dataclasses import fields
import os
from os.path import join
from copy import deepcopy
import time

import drjit as dr
import mitsuba as mi
from tqdm import tqdm
from batched import render_batch, render_batch_voxel, render_batch_l2loss
from util import save_params, get_single_medium, calculate_psnr
import numpy as np
import scipy.ndimage as ndi
import json
import losses


def save_opt_state(path, opt):
    state_np = {}

    for k, (m, v) in opt.state.items():
        state_np[k] = (
            np.array(dr.detach(m).numpy()),
            np.array(dr.detach(v).numpy()),
        )

    ckpt = {
        "state": state_np,  # numpy arrays
        "t": dict(opt.t),         # defaultdict[int]
    }

    with open(path, "wb") as f:
        pickle.dump(ckpt, f)

def sync():
    dr.eval()      # ensure Dr.Jit kernel launch
    dr.sync_thread()

def load_opt_state(path, opt, params):
    """
    path   : checkpoint path
    opt    : mi.ad.Adam (freshly created)
    params : mi.traverse(scene) result
    """
    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    # ---- restore t ----
    opt.t.clear()
    for k, v in ckpt["t"].items():
        opt.t[k] = int(v)

    original_state = opt.state

    # ---- restore Adam moments ----
    for k, (m_np, v_np) in ckpt["state"].items():
        if k not in params:
            print(f"[warn] param {k} not in current params, skipping")
            continue

        m = original_state[k][0] * 0 + m_np
        v = original_state[k][1] * 0 + v_np

        opt.state[k] = (m, v)
    # import ipdb; ipdb.set_trace()
    print(f'[i] finish Loading optimizer state from {path}')


def load_scene(scene_config, reference=False, **kwargs):
    scene_vars = scene_config.ref_scene_vars if reference else scene_config.normal_scene_vars
    if reference and scene_config.ref_fname:
        scene = mi.load_file(scene_config.ref_fname, **scene_vars, **kwargs)
    else:
        scene = mi.load_file(scene_config.fname, **scene_vars, **kwargs)

    return scene


def render_reference_image_default(
    scene_config,
    to_render,
    seed=1234,
    max_rays_per_pass=720*720*2048,
):
    import integrators
    scene = load_scene(scene_config, reference=True)
    integrator = mi.load_dict({
        'type': scene_config.ref_integrator,
        'max_depth': scene_config.max_depth,
    })
    ref_spp = scene_config.ref_spp

    def _render_to_paths(paths_map):
        for s, fname in tqdm(paths_map.items(), desc=f'Reference renderings @ {ref_spp}'):
            sensor = scene.sensors()[s]

            total_rays = dr.prod(sensor.film().size()) * ref_spp
            pass_count = int(dr.ceil(total_rays / max_rays_per_pass))
            spp_per_pass = int(dr.ceil(ref_spp / pass_count))
            assert spp_per_pass * pass_count >= ref_spp

            result = None
            for pass_i in tqdm(range(pass_count), desc='Render passes', leave=False):
                image = mi.render(
                    scene,
                    sensor=sensor,
                    integrator=integrator,
                    spp=spp_per_pass,
                    seed=seed + pass_i
                )
                dr.eval(image)
                result = image / pass_count if result is None else result + image / pass_count
                del image

            mi.Bitmap(result).write(fname)

    _render_to_paths(to_render)


def render_reference_image_with_envmap_rot(
    scene_config,
    to_render,
    seed=1234,
    max_rays_per_pass=720*720*2048,
    envmap_rotations=None,
    include_axis_rotations=False,
    subfolder_prefix='envrot',
    copy_default=True,   # True = shutil.copy2, False = os.rename
):
    import integrators
    import shutil

    scene = load_scene(scene_config, reference=True)
    integrator = mi.load_dict({
        'type': scene_config.ref_integrator,
        'max_depth': scene_config.max_depth,
    })
    ref_spp = scene_config.ref_spp

    params = mi.traverse(scene)

    # --------------------------------------------------
    # Find envmap to_world
    # --------------------------------------------------
    env_keys = [
        k for k in params.keys()
        if k.endswith('.to_world') and 'EnvironmentMapEmitter' in k
    ]
    env_to_world_key = env_keys[0] if env_keys else 'EnvironmentMapEmitter.to_world'
    if env_to_world_key not in params:
        print('[!] EnvironmentMapEmitter.to_world not found; skip env rotations.')
        return

    # --------------------------------------------------
    # Render helper
    # --------------------------------------------------
    def _render_to_paths(paths_map):
        for s, fname in tqdm(paths_map.items(), desc=f'Reference renderings @ {ref_spp}'):
            sensor = scene.sensors()[s]
            total_rays = dr.prod(sensor.film().size()) * ref_spp
            pass_count = int(dr.ceil(total_rays / max_rays_per_pass))
            spp_per_pass = int(dr.ceil(ref_spp / pass_count))

            result = None
            for pass_i in tqdm(range(pass_count), desc='Render passes', leave=False):
                image = mi.render(
                    scene,
                    sensor=sensor,
                    integrator=integrator,
                    spp=spp_per_pass,
                    seed=seed + pass_i
                )
                dr.eval(image)
                result = image / pass_count if result is None else result + image / pass_count
                del image

            mi.Bitmap(result).write(fname)

    def _transform_to_list(t):
        # Ensure the transform is converted into a plain Python list for JSON.
        tf = t if isinstance(t, mi.Transform4f) else mi.Transform4f(t)
        # Convert via NumPy to drop Dr.Jit types and ensure JSON-serializable floats.
        mat = np.array(tf.matrix, dtype=float).reshape(4, 4)
        return mat.tolist()

    # --------------------------------------------------
    # Build rotation list
    # --------------------------------------------------
    rotations_named = [
        ('identity', mi.Transform4f.rotate(mi.Vector3f(0, 1, 0), 0.0))
    ]

    if include_axis_rotations:
        rotations_named.extend([
            ('negz', mi.Transform4f.rotate(mi.Vector3f(0, 1, 0), 180.0)),
            ('posx', mi.Transform4f.rotate(mi.Vector3f(0, 1, 0), -90.0)),
            ('negx', mi.Transform4f.rotate(mi.Vector3f(0, 1, 0), 90.0)),
            ('posy', mi.Transform4f.rotate(mi.Vector3f(1, 0, 0), 90.0)),
            ('negy', mi.Transform4f.rotate(mi.Vector3f(1, 0, 0), -90.0)),
        ])

    if envmap_rotations:
        for i, rot in enumerate(envmap_rotations):
            name = f'user{i:02d}'
            rot_tf = rot if isinstance(rot, mi.Transform4f) else mi.Transform4f(rot)
            rotations_named.append((name, rot_tf))

    # --------------------------------------------------
    # Default reference detection
    # --------------------------------------------------
    base_ref_dir = scene_config.references
    default_paths = to_render

    def _default_exists():
        # Consider defaults available only if a non-empty map is provided
        # and all files exist.
        return bool(default_paths) and all(os.path.exists(p) for p in default_paths.values())

    default_available = _default_exists()
    print(f'[i] Default reference images available: {default_available}')

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for idx, (rot_name, rot_tf) in enumerate(rotations_named):
        sub_name = f"{subfolder_prefix}_{idx:02d}_{rot_name}"
        sub_dir = join(base_ref_dir, sub_name)
        os.makedirs(sub_dir, exist_ok=True)

        # save rotation json
        with open(join(sub_dir, 'rotation.json'), 'w') as f:
            json.dump({
                'name': rot_name,
                'matrix': _transform_to_list(rot_tf)
            }, f, indent=2)

        rot_paths = {
            s: join(sub_dir, f'ref_{s:06d}.exr')
            for s in scene_config.sensors
        }

        # --------------------------------------------------
        # identity: reuse default if possible
        # --------------------------------------------------
        if rot_name == 'identity' and default_available:
            print('[✓] Reusing default reference for identity envmap')
            for s, src in default_paths.items():
                dst = rot_paths[s]
                if not os.path.exists(dst):
                    if copy_default:
                        shutil.copy2(src, dst)
                    else:
                        os.rename(src, dst)
            continue

        # --------------------------------------------------
        # Otherwise render
        # --------------------------------------------------
        params[env_to_world_key] = rot_tf
        params.update()
        _render_to_paths(rot_paths)




def get_reference_image_paths(scene_config, overwrite=False, use_rot_envmap=False):
    ref_dir = scene_config.references
    os.makedirs(ref_dir, exist_ok=True)

    fname_pattern = join(ref_dir, 'ref_{:06d}.exr')
    paths = { s: fname_pattern.format(s) for s in scene_config.sensors }

    if use_rot_envmap:
        import re
        # When using rotated envmaps, ensure the six axis-rotation
        # reference sets exist. Do not render default refs (identity
        # is already covered by one rotation set).
        # If currently inside a rotation subfolder, operate on its parent.
        base_dir = ref_dir
        basename = os.path.basename(ref_dir)
        if re.match(r'^envrot_\d{2}_[a-z]+$', basename):
            base_dir = os.path.dirname(ref_dir)

        # Must match the naming in render_reference_image_with_envmap_rot
        suffix = ['identity', 'negz', 'posx', 'negx', 'posy', 'negy']
        missing_any = bool(overwrite)
        missing_by_subdir = {}
        for i, suf in enumerate(suffix):
            sub_dir = join(base_dir, f'envrot_{i:02d}_{suf}')
            sub_missing = []
            for s in scene_config.sensors:
                if not os.path.isfile(join(sub_dir, f'ref_{s:06d}.exr')):
                    sub_missing.append(s)
            if sub_missing:
                missing_any = True
                missing_by_subdir[sub_dir] = sub_missing

        if missing_any:
            print(f"[i] Missing rotated reference images detected ({list(missing_by_subdir.keys()) or 'all'}); rendering axis rotations…")
            # Temporarily set references to base_dir and render ONLY rotations
            # by providing an empty to_render map and enabling axis rotations.
            original_ref_dir = scene_config.references
            try:
                scene_config.references = base_dir
                render_reference_image_with_envmap_rot(scene_config, to_render={}, include_axis_rotations=True)
            finally:
                scene_config.references = original_ref_dir
        # Return mapping for current ref_dir (may be a rotation subdir).
        return paths

    else:
        # Default path: validate and (re)render the non-rotated references
        if overwrite:
            missing_paths = deepcopy(paths)
        else:
            missing_paths = { s: fname for s, fname in paths.items()
                            if not os.path.isfile(fname) }
        if missing_paths:
            print(f'[i] Missing reference images detected ({missing_paths}); rendering...')
            render_reference_image_default(scene_config, missing_paths)
    return paths
    

def get_preview_image_paths(scene_config):
    ref_dir = scene_config.references
    fname_pattern = join(ref_dir, 'ref_{:06d}.exr')
    paths = { s: fname_pattern.format(s) for s in scene_config.preview_sensors }

    for s, fname in paths.items():
        if not os.path.isfile(fname):
            raise FileNotFoundError(f'Preview reference image not found: {fname}')

    return paths

def load_reference_images(paths, batchify=False):
    if batchify:
        import numpy as np
        # Note: we rely on `paths` being ordered consistently.
        batched = np.concatenate([
            np.array(mi.Bitmap(f))[None, ...]
            for _, f in paths.items()
        ], axis=0)
        return mi.TensorXf(batched)
    else:
        return {
            s: mi.TensorXf(mi.Bitmap(f))
            for s, f in paths.items()
        }


def gather_ref_values(ref_images, sensor_idx, pixel_idx):
    # Shape: image_idx * height * width * channels
    sh = dr.shape(ref_images)
    assert len(sh) == 4
    channels = sh[-1]
    assert channels in (3, 4)
    sh = sh[:3]
    indices = (
        sensor_idx * dr.prod(sh[1:])
        + pixel_idx.y * dr.prod(sh[2:])
        + pixel_idx.x * dr.prod(sh[3:])
    )
    color_type = mi.Vector3f if channels == 3 else mi.Vector4f

    values = dr.gather(color_type, ref_images.array, indices)

    # Shape: height * batch_size * channels
    return mi.TensorXf(dr.ravel(values), shape=(1, dr.width(pixel_idx), channels))


def render_previews(output_dir, opt_config, scene_config, scene, integrator, it_i, no_bg = False,  write=True, spp=None):
    if it_i == 'initial':
        if not opt_config.render_initial:
            return
        suffix = '_init'
    elif it_i == 'final':
        if not opt_config.render_final:
            return
        suffix = '_final'
    elif isinstance(it_i, int):
        suffix = f'_{it_i:08d}'
    else:
        assert isinstance(it_i, str)
        suffix = it_i

    preview_spp = opt_config.preview_spp or opt_config.spp

    image_list = []
    fmt = getattr(opt_config, 'preview_format', 'exr')
    free_thresh = getattr(opt_config, 'preview_free_threshold_gb', 0.0)
    ext = '.png' if fmt == 'png' else '.exr'
    for s in scene_config.preview_sensors:
        bg_tag = '_nobg' if no_bg else ''
        actual_spp = preview_spp if spp is None else spp
        fname = join(output_dir, f'opt{suffix}{bg_tag}_{s:04d}_spp_{actual_spp}{ext}')
        if spp is None:
            image = mi.render(scene, integrator=integrator, sensor=s,
                            seed=1234, spp=preview_spp)
        else:
            if spp <= 1024:
                image = mi.render(scene, integrator=integrator, sensor=s,
                            seed=1234, spp=spp)
            else:
                image = 0
                for render_i in range(spp // 1024):
                    partial_image = mi.render(scene, integrator=integrator, sensor=s,
                                seed=1234 + render_i, spp=1024)
                    image += partial_image
                image /= (spp // 1024)
        if write:
            if fmt == 'png':
                from util import write_preview_png
                write_preview_png(image, fname, free_threshold_gb=free_thresh)
            else:
                # EXR (legacy). free-threshold gate also applied if set.
                if free_thresh <= 0 or _free_gb_safe(os.path.dirname(fname)) >= free_thresh:
                    mi.Bitmap(image).write(fname)
        image_list.append(image)

    return image_list


def _free_gb_safe(path):
    try:
        os.makedirs(path, exist_ok=True)
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize / (1024 ** 3)
    except OSError:
        return 0.0


def initialize_scene(opt_config, scene_config, scene):
    params = mi.traverse(scene)
    params.keep(scene_config.param_keys)

    # Set params to their initial values
    for k, v in scene_config.start_from_value.items():
        assert k in params
        if v is None:
            assert not opt_config.upsample
            continue

        # --- Upsampling support
        # If parameter resolutions will be upsampled during the optimization,
        # figure out the initial resolution that will lead to the correct
        # final resolution after n upsampling steps.
        shape = dr.shape(params[k])
        if opt_config.upsample:
            assert len(shape) == 4
            upsample_res_factor = 2 ** len(opt_config.upsample)
            # Preserve channel count
            init_res = (*[max(1, s // upsample_res_factor) for s in shape[:3]], shape[-1])
            if 1 in init_res[:3]:
                raise ValueError(f'Initial resolution not supported: {init_res}. Maybe reduce upsample_steps?')

            if '.sigma_t.' in k:
                adjust_majorant_res_factor(scene_config, scene, init_res)
        else:
            init_res = shape

        params[k] = type(params[k])(v, shape=init_res)

    params.update()
    return params


def enforce_valid_params(scene_config, opt):
    """Projects parameters back to their legal range."""
    for k, v in opt.items():
        if k.endswith('sigma_t.data'):
            opt[k] = dr.clip(v, 0, scene_config.max_density)
        elif k.endswith('emission.data'):
            opt[k] = dr.maximum(v, 0)
        elif k.endswith('albedo.data'):
            opt[k] = dr.clip(v, 0, 1)
        else:
            raise ValueError


def adjust_majorant_res_factor(scene_config, scene, density_res):
    res_factor = scene_config.majorant_resolution_factor

    if res_factor > 1:
        min_side = dr.min(density_res[:3])
        # For the current density res, find the largest factor that
        # results in a meaningful supergrid resolution.
        while (res_factor > 1) and (min_side // res_factor) < 4:
            res_factor -= 1
    # Otherwise, just disable the supergrid.
    if res_factor <= 1:
        res_factor = 0

    medium = get_single_medium(scene)
    current = medium.majorant_resolution_factor()
    if current != res_factor:
        medium.set_majorant_resolution_factor(res_factor)
        print(f'[i] Updated majorant supergrid resolution factor: {current} → {res_factor}')



def upsample_grid(current_values, old_res, new_res):
    """Upsample values of a 3D grid using first order interpolation."""
    from scipy.ndimage import zoom
    assert isinstance(current_values, (mi.TensorXf, dr.detached_t(mi.TensorXf))), \
            'Unsupported type for upsampling: {}'.format(type(current_values))
    if tuple(old_res) == tuple(new_res):
        return mi.TensorXf(dr.detach(current_values))

    n_channels = old_res[-1]
    assert len(old_res) == 4 and len(new_res) == 4
    assert new_res[-1] == n_channels

    # TODO: replace with a pure DrJit-based solution
    factors = [r1 / r2 for r1, r2 in zip(new_res, old_res)]
    current_values = current_values.numpy()
    if current_values.ndim == 3:
        current_values = current_values[..., None]

    new_values = zoom(current_values, factors, order=1,
                      mode='nearest', prefilter=False, grid_mode=True)
    new_values = mi.TensorXf(new_values)
    assert new_values.shape == new_res, (str(new_values.shape), str(new_res))
    return new_values


def upsample_params_if_needed(opt_config, scene_config, scene, params, opt, it_i):
    if not opt_config.should_upsample(it_i):
        return False

    majorant_res_factor = scene_config.majorant_resolution_factor

    for k in scene_config.param_keys:
        v = opt[k]
        old_res = dr.shape(v)
        assert len(old_res) == 4
        new_res = (*[2 * r for r in old_res[:3]], old_res[-1])

        # opt[k] = dr.upsample(v, shape=new_res)
        opt[k] = upsample_grid(v, old_res, new_res)

        assert dr.shape(opt[k]) == new_res
        print(f'[i] Upsampled parameter "{k}" at iteration {it_i}: {old_res} → {new_res}')

        if '.sigma_t.' in k:
            adjust_majorant_res_factor(scene_config, scene, new_res)

    medium = get_single_medium(scene)
    medium.set_majorant_resolution_factor(majorant_res_factor)
    params.update(opt)
    return True


def create_checkpoint(output_dir, opt_config, scene_config, params, name_or_it, opt=None):
    prefix = name_or_it
    if name_or_it == 'initial':
        if not opt_config.checkpoint_initial:
            return
    elif name_or_it == 'final':
        if not opt_config.checkpoint_final:
            return
    elif isinstance(name_or_it, int):
        if (name_or_it == 0) or (not opt_config.checkpoint_stride) or (name_or_it % opt_config.checkpoint_stride) != 0:
            return
        prefix = f'{name_or_it:08d}'
    else:
        raise ValueError('Unsupported: ' + str(name_or_it))

    checkpoint_dir = join(output_dir, 'params')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_params(checkpoint_dir, scene_config, params, prefix,
                skip_emission=opt_config.skip_emission,
                compression=opt_config.save_compression,
                async_compression=opt_config.async_compression)
    if opt is not None and opt_config.save_opt_state:
        save_opt_state(join(checkpoint_dir, f'opt_state_{prefix}.pkl'), opt)


def get_importance_grid(opt_config, scene, params, voxel_size, sensors_dr, film_size, pixel_format, batch_film, batch_render_sampler, integrator, output_dir):
    # batch_size = voxel_num
    nx, ny, nz = voxel_size
    batch_size = nx * ny * nz
    image_lst = []
    # --- Main optimization loop
    for it_i in tqdm(range(opt_config.importance_iter), desc='Optimization',
                     dynamic_ncols=True):
        seed, _ = mi.sample_tea_32(2 * it_i + 0, opt_config.base_seed)
        seed_grad, _ = mi.sample_tea_32(2 * it_i + 1, opt_config.base_seed)
        # upsample_params_if_needed(opt_config, scene_config, scene, params, opt, it_i)


        # nx, ny, nz = 32, 32, 32

        # shape = (nx, ny, nz)
        # coords = np.indices(shape)                   # shape -> (3, 32, 32, 32)
        # grid = coords.transpose(3, 2, 1, 0)
        # # flat = grid.reshape(-1, 3)
        # # import ipdb;ipdb.set_trace()
        # # dr_grid = dr.cuda.ad.TensorXf(grid.astype(np.float32))  # shape -> (32, 32, 32, 3)
        # # import ipdb; ipdb.set_trace()
        # params['medium1.emission.data'] = dr.cuda.ad.TensorXf(grid.astype(np.float32))
        # params.update()
        with dr.suspend_grad():
            image, batch_film, batch_render_sampler, _, _ = render_batch_voxel(
                batch_size, scene, sensors_dr, film_size,
                params=params, integrator=integrator, film=batch_film,
                pixel_format=pixel_format, sampler=batch_render_sampler,
                spp=1, spp_grad=1,
                seed=seed, seed_grad=seed_grad
            )
        # loss_value = opt_config.loss(image, ref_values)
        image_lst.append(image.numpy().reshape(nx,ny,nz,-1))
    inscattering_radiance = np.array(image_lst)
    # inscattering_radiance = inscattering_radiance[:,0,:,:]
    # compute mean, variance and second moment along the first axis
    inscattering_radiance_mean = np.mean(inscattering_radiance, axis=0)
    inscattering_radiance_var = np.var(inscattering_radiance, axis=0)
    # inscattering_radiance_second_moment = inscattering_radiance_var + inscattering_radiance_mean**2
    # np.savez_join(output_dir, 'inscattering_radiance_stats.npz',
    #              mean=inscattering_radiance_mean,
    #              variance=inscattering_radiance_var,
    #              second_moment=inscattering_radiance_second_moment)
    # import ipdb; ipdb.set_trace()

    # upsample to scene emission resolution
    resolution = params['medium1.emission.data'].numpy().shape

    zoom_factor = (resolution[0]/nx, resolution[1]/ny, resolution[2]/nz, 1)
    up_grid_var = ndi.zoom(inscattering_radiance_var, zoom=zoom_factor, order=1)
    up_grid_mean = ndi.zoom(inscattering_radiance_mean, zoom=zoom_factor, order=1)

    # import ipdb; ipdb.set_trace()

    # write out volume grids for debugging
    print(inscattering_radiance_mean.shape)
    mean_grid = mi.VolumeGrid(up_grid_mean)
    fname = join(output_dir, f'inscattering_radiance_mean_iter{it_i:08d}.vol')
    mean_grid.write(fname)
    var_grid = mi.VolumeGrid(up_grid_var)
    fname = join(output_dir, f'inscattering_radiance_var_iter{it_i:08d}.vol')
    var_grid.write(fname)

    return up_grid_mean, up_grid_var

def run_optimization(output_dir, opt_config, scene_config, int_config, use_log=False, use_rot_envmap=False):

    if use_log:
        import wandb
        # initialize wandb run
        wandb.init(project=WANDB_EXP_NAME, name=opt_config.name)
        
        # log config parameters
        for f in fields(opt_config):
            wandb.config[f.name] = getattr(opt_config, f.name)
        wandb.config["scene"] = scene_config.name
        wandb.config["integrator"] = int_config.name
    import integrators
    print(f'[i] Starting optimization:')
    print(f'    Scene:      {scene_config.name}')
    print(f'    Integrator: {int_config.name}')
    print(f'    Output dir: {output_dir}')
    print(f'    Opt params:')
    for f in fields(opt_config):
        print(f'        {f.name}: {opt_config.__dict__[f.name]}')


    batch_size = opt_config.batch_size
    if use_rot_envmap:
        # Must match the naming in render_reference_image_with_envmap_rot
        suffix = ['identity', 'negz', 'posx', 'negx', 'posy', 'negy']

        # Ensure all rotated reference sets (and rotation.json files) exist.
        _ = get_reference_image_paths(scene_config, use_rot_envmap=True)

        group_index = list(scene_config.envmap_rot_group)
        if not group_index:
            raise ValueError('envmap_rot_group is empty while use_rot_envmap=True')

        envmap_rotation_dict = {}
        ref_images_dict = {}
        scene_config_references_parent = scene_config.references

        # Prepare references and rotation matrices for each required group index
        for g_idx in tqdm(sorted(set(group_index))):
            sub_dir = join(scene_config_references_parent, f"envrot_{g_idx:02d}_{suffix[g_idx]}")

            # Load or render references for this rotation set
            scene_config.references = sub_dir
            ref_paths = get_reference_image_paths(scene_config, use_rot_envmap=True)
            ref_images = load_reference_images(ref_paths, batchify=(batch_size is not None))
            ref_images_dict[g_idx] = ref_images

            rot_json_path = join(sub_dir, 'rotation.json')
            if not os.path.isfile(rot_json_path):
                raise FileNotFoundError(f'Rotation json not found after rendering: {rot_json_path}')
            with open(rot_json_path, 'r') as f:
                rot_data = json.load(f)
                envmap_rotation_dict[g_idx] = np.array(rot_data['matrix'])

            print(f'[i] Loaded env rotation set {g_idx}: {sub_dir}')

        # Restore original references and pick the default ref images for the first group
        scene_config.references = scene_config_references_parent
        ref_images = ref_images_dict[group_index[0]]
        print('envmap rotation group index: ', group_index)
    else:
        ref_paths = get_reference_image_paths(scene_config)
        ref_images = load_reference_images(ref_paths, batchify=(batch_size is not None))

    # Load preview ref images separately — keyed by sensor ID (dict, not
    # batchified).  This decouples preview PSNR from the training sensor list,
    # allowing preview_sensors to be test views not in scene_config.sensors.
    preview_ref_paths = get_preview_image_paths(scene_config)
    preview_ref_images = load_reference_images(preview_ref_paths, batchify=False)

    if scene_config.load_ckpt_dir is not None:
        checkpoint_dir = join(scene_config.load_ckpt_dir, 'params')
        checkpoint_name = f'{scene_config.load_ckpt_iter:08d}'
        print(f'[i] Loading checkpoint from {checkpoint_dir}, iteration {checkpoint_name}...')
        scene_config.normal_scene_vars['medium_filename'] = join(checkpoint_dir, f'{checkpoint_name}-medium1_sigma_t.vol')
        scene_config.normal_scene_vars['albedo_filename'] = join(checkpoint_dir, f'{checkpoint_name}-medium1_albedo.vol')
        scene_config.start_from_value['medium1.sigma_t.data'] = None
        scene_config.start_from_value['medium1.albedo.data'] = None
    scene = load_scene(scene_config, reference=False)
    init_params = mi.traverse(scene)

    integrator = int_config.create(max_depth=scene_config.max_depth)
    # Background-free preview integrator: a separately-registered forward
    # integrator with hide_emitters set at construction (via props), rather than
    # mutating an existing instance's attribute (more robust under Mitsuba's
    # plugin-registration mechanism).
    no_bg_integrator = mi.load_dict({
        'type': 'volpathsimple_no_bg',
        'max_depth': scene_config.max_depth,
        'rr_depth': scene_config.max_depth,
        'use_nee': True,
        'hide_emitters': True,
    })
    sampler = mi.scalar_rgb.PCG32(initstate=93483)

    n_sensors = len(scene_config.sensors)
    spp_grad = opt_config.spp
    spp_primal = spp_grad * opt_config.primal_spp_factor

    if batch_size is not None:
        sensors_dr = dr.gather(mi.SensorPtr, scene.sensors_dr(),
                               mi.UInt32(scene_config.sensors))
        # Assume that all sensors have the same dimensions
        first_film = scene.sensors()[scene_config.sensors[0]].film()
        film_size = first_film.crop_size()
        pixel_format = (
            mi.Bitmap.PixelFormat.RGBA if mi.has_flag(first_film.flags(), mi.FilmFlags.Alpha)
            else mi.Bitmap.PixelFormat.RGB
        )
        batch_film = None
        batch_render_sampler = None
        del first_film

    # --- Initialization
    if scene_config.load_ckpt_dir is not None and scene_config.load_ckpt_iter is not None:
        params = mi.traverse(scene)
        params.keep(scene_config.param_keys)
        opt = opt_config.optimizer(params)
        opt_state_path = join(checkpoint_dir, f'opt_state_{checkpoint_name}.pkl')
        if os.path.isfile(opt_state_path):
            load_opt_state(opt_state_path, opt, params)
            print(f'[i] Loaded optimizer state from {opt_state_path}')
        else:
            print(f'[i] No opt state pkl at {opt_state_path}; starting Adam moments from 0 (volumes still loaded from ckpt)')
    else:
        params = initialize_scene(opt_config, scene_config, scene)
        opt = opt_config.optimizer(params)

    for _, v in params.items():
        dr.enable_grad(v)

    if scene_config.load_ckpt_dir is None:
        create_checkpoint(output_dir, opt_config, scene_config, params, 'initial')
    # Always render the initial preview so loaded checkpoints can be verified
    render_previews(output_dir, opt_config, scene_config, scene, integrator, 'initial')
    # Write out the reference images corresponding to the previews for easy comparison
    for s in scene_config.preview_sensors:
        fname = join(output_dir, f'ref_{s:04d}.exr')
        mi.Bitmap(preview_ref_images[s]).write(fname)
    # Print initial-preview PSNR vs ref (useful for verifying loaded ckpts)
    try:
        psnr_init_sum = 0.0
        n_init = 0
        preview_spp_eff = opt_config.preview_spp or opt_config.spp
        for s in scene_config.preview_sensors:
            init_exr = join(output_dir, f'opt_init_{s:04d}_spp_{preview_spp_eff}.exr')
            if os.path.isfile(init_exr):
                rendered = np.array(mi.Bitmap(init_exr))
                ref_np = preview_ref_images[s].numpy() if hasattr(preview_ref_images[s], 'numpy') else np.array(preview_ref_images[s])
                if rendered.shape[-1] != ref_np.shape[-1]:
                    ch = min(rendered.shape[-1], ref_np.shape[-1])
                    rendered = rendered[..., :ch]
                    ref_np = ref_np[..., :ch]
                psnr_v = calculate_psnr(ref_np, rendered)
                print(f'[initial-preview] sensor={s} PSNR={psnr_v:.2f} dB')
                psnr_init_sum += psnr_v
                n_init += 1
        if n_init > 0:
            print(f'[initial-preview] mean PSNR = {psnr_init_sum/n_init:.2f} dB ({n_init} sensors)')
    except Exception as _e:
        print(f'[initial-preview] PSNR compute skipped: {_e}')

    total_time = 0.0
    # --- Main optimization loop
    if scene_config.load_ckpt_iter is not None:
        start_iter = scene_config.load_ckpt_iter+1
    else:
        start_iter = 0

    for it_i in tqdm(range(start_iter, opt_config.n_iter), desc='Optimization',
                          dynamic_ncols=True):
        # Early exit for partial-training runs (e.g. relight convergence
        # captures) — keeps upsampling schedule aligned with n_iter=8000
        # but stops saving checkpoints after max_iter.
        if opt_config.max_iter is not None and it_i > opt_config.max_iter:
            break
        # random choose a envmap rotation for this iteration
        if use_rot_envmap:
            g_idx = scene_config.envmap_rot_group[it_i % len(scene_config.envmap_rot_group)]
            init_params['EnvironmentMapEmitter.to_world'] = mi.Transform4f(envmap_rotation_dict[g_idx])
            init_params.update()
            ref_images = ref_images_dict[g_idx]

        sync()
        start_i = time.perf_counter()
        seed, _ = mi.sample_tea_32(2 * it_i + 0, opt_config.base_seed)
        seed_grad, _ = mi.sample_tea_32(2 * it_i + 1, opt_config.base_seed)
        opt.set_learning_rate(opt_config.learning_rates(scene_config, it_i))
        upsample_params_if_needed(opt_config, scene_config, scene, params, opt, it_i)
        # if opt_config.importance_iter_stride is not None and it_i % opt_config.importance_iter_stride == 0:
        #     mean_grid, var_grid = get_importance_grid(opt_config, scene, params, [32,32,32], sensors_dr, film_size, pixel_format, batch_film, batch_render_sampler, integrator, output_dir)
        
        if batch_size is not None:
            # --- Batched rendering
            image, batch_film, batch_render_sampler, sensor_idx, pixel_idx = render_batch(
                batch_size, scene, sensors_dr, film_size,
                params=params, integrator=integrator, film=batch_film,
                pixel_format=pixel_format, sampler=batch_render_sampler,
                spp=spp_primal, spp_grad=spp_grad,
                seed=seed, seed_grad=seed_grad
            )
            ref_values = gather_ref_values(ref_images, sensor_idx, pixel_idx)

            if opt_config.loss is losses.half_l2:
                with dr.suspend_grad():
                    # Re-render primal only for loss computation
                    primal_image, _, _, _, _ = render_batch_l2loss(
                        batch_size, scene, sensors_dr, film_size,
                        params=params, integrator=integrator, film=batch_film,
                        pixel_format=pixel_format, sampler=batch_render_sampler,
                        spp=spp_primal, spp_grad=spp_grad,
                        seed=seed, seed_grad=seed_grad
                    )
           
        else:
            # --- Sensor-based rendering
            sensor_i = scene_config.sensors[int(sampler.next_float32() * n_sensors)]
            image = mi.render(scene, params=params, integrator=integrator, sensor=sensor_i,
                              spp=spp_primal, spp_grad=spp_grad,
                              seed=seed, seed_grad=seed_grad)
            ref_values = ref_images[sensor_i]
        if opt_config.loss is losses.half_l2:
            loss_value = opt_config.loss(image, primal_image, ref_values)
        else:
            loss_value = opt_config.loss(image, ref_values)
        dr.backward(loss_value)

        opt.step()
        enforce_valid_params(scene_config, opt)
        params.update(opt)
        sync()
        end_i = time.perf_counter()
        total_time += (end_i - start_i)
        create_checkpoint(output_dir, opt_config, scene_config, params, it_i, opt)
        # albeod_mean = params['medium1.albedo.data'].numpy().mean()
        # print("albedo mean: ", albeod_mean)
        with dr.suspend_grad():
            if (it_i > 0 and it_i % opt_config.preview_stride == 0) or (it_i > 0 and it_i < 320 and it_i % 10 == 0):
                _wp = opt_config.write_preview_exrs
                image_lst = render_previews(output_dir, opt_config, scene_config, scene, integrator, it_i, write=_wp)
                # also write a no-background preview (return value unused; written to disk as a side effect)
                render_previews(output_dir, opt_config, scene_config, scene, no_bg_integrator, it_i, no_bg=True, write=_wp)
                # save ref (skip if previews are disabled — saves a lot of disk during video runs)
                if _wp:
                    fmt = getattr(opt_config, 'preview_format', 'exr')
                    free_thresh = getattr(opt_config, 'preview_free_threshold_gb', 0.0)
                    for s in scene_config.preview_sensors:
                        if fmt == 'png':
                            from util import write_preview_png
                            write_preview_png(preview_ref_images[s],
                                              join(output_dir, f'ref_{s:04d}_it_{it_i:04d}.png'),
                                              free_threshold_gb=free_thresh)
                        else:
                            fname = join(output_dir, f'ref_{s:04d}_it_{it_i:04d}.exr')
                            if free_thresh <= 0 or _free_gb_safe(os.path.dirname(fname)) >= free_thresh:
                                mi.Bitmap(preview_ref_images[s]).write(fname)
            if (it_i > 0) and use_log:
                # --- routine metric logging
                with dr.suspend_grad():
                    primal_l1_loss = dr.mean(dr.abs(image - ref_values))

                # assemble the metrics dict
                metrics = {
                    "loss": float(loss_value.numpy()),
                    "primal_l1_loss": float(primal_l1_loss.numpy()),
                    "step": it_i,
                    "opt_time": total_time
                }

                # learning-rate logging
                lr_dict = opt_config.learning_rates(scene_config, it_i)
                for name, value in lr_dict.items():
                    metrics[f"learning_rate/{name}"] = float(value)

                # log PSNR once every preview_stride iterations
                if (it_i > 0 and it_i % opt_config.preview_stride == 0) or (it_i > 0 and it_i < 320 and it_i % 10 == 0):
                    # Build sensor_id → image_lst index mapping
                    sensor_to_idx = {s: si for si, s in enumerate(scene_config.preview_sensors)}

                    # psnr_train: sensors ∈ training set (comparable to paper wandb)
                    train_sensors = scene_config.preview_train_sensors
                    if train_sensors:
                        psnr_train = sum(
                            calculate_psnr(preview_ref_images[s].numpy(), image_lst[sensor_to_idx[s]].numpy())
                            for s in train_sensors
                        ) / len(train_sensors)
                        metrics["psnr_train"] = psnr_train
                    else:
                        psnr_train = None

                    # psnr_test: sensors ∉ training set (held-out, clean eval)
                    test_sensors = scene_config.preview_test_sensors
                    if test_sensors:
                        psnr_test = sum(
                            calculate_psnr(preview_ref_images[s].numpy(), image_lst[sensor_to_idx[s]].numpy())
                            for s in test_sensors
                        ) / len(test_sensors)
                        metrics["psnr_test"] = psnr_test
                    else:
                        psnr_test = None

                    # psnr: union average (backward compat); prefer train if test absent
                    psnr_mean = 0
                    for si, s in enumerate(scene_config.preview_sensors):
                        psnr_mean += calculate_psnr(preview_ref_images[s].numpy(), image_lst[int(si)].numpy())
                    psnr = psnr_mean / len(scene_config.preview_sensors)
                    metrics["psnr"] = psnr

                    train_str = f'{psnr_train:.2f}' if psnr_train is not None else 'N/A'
                    test_str = f'{psnr_test:.2f}' if psnr_test is not None else 'N/A'
                    print(f'PSNR: {psnr:.2f} dB  (train={train_str}  test={test_str})')

                    # --- Preview image loss split by train / test ---
                    def _img_losses(sensor_list):
                        l1=0; l2=0; rl1=0; rl2=0
                        for s in sensor_list:
                            si = sensor_to_idx[s]
                            opt_img = image_lst[int(si)].numpy()
                            ref_img = preview_ref_images[s].numpy()
                            diff = opt_img - ref_img
                            _l1 = float(np.mean(np.abs(diff)))
                            _l2 = float(np.mean(diff * diff))
                            _ref_l1 = float(np.mean(np.abs(ref_img))) + 1e-12
                            _ref_l2 = float(np.mean(ref_img * ref_img)) + 1e-12
                            l1 += _l1; l2 += _l2
                            rl1 += _l1 / _ref_l1; rl2 += _l2 / _ref_l2
                        n = len(sensor_list)
                        return l1/n, l2/n, rl1/n, rl2/n

                    if train_sensors:
                        tl1, tl2, trl1, trl2 = _img_losses(train_sensors)
                        metrics["preview_l1_loss"] = tl1
                        metrics["preview_l2_loss"] = tl2
                        metrics["preview_rell1_loss"] = trl1
                        metrics["preview_rell2_loss"] = trl2
                    if test_sensors:
                        el1, el2, erl1, erl2 = _img_losses(test_sensors)
                        metrics["preview_test_l1_loss"] = el1
                        metrics["preview_test_l2_loss"] = el2
                        metrics["preview_test_rell1_loss"] = erl1
                        metrics["preview_test_rell2_loss"] = erl2
                    print(f'Preview Loss: L1={metrics.get("preview_l1_loss",float("nan")):.6f}  '
                          f'relL1={metrics.get("preview_rell1_loss",float("nan")):.6f}  '
                          f'L2={metrics.get("preview_l2_loss",float("nan")):.6f}  '
                          f'relL2={metrics.get("preview_rell2_loss",float("nan")):.6f}')

                # write to wandb
                # Keep WandB step aligned with resumed iteration index
                wandb.log(metrics, step=it_i)

    create_checkpoint(output_dir, opt_config, scene_config, params, 'final')
    render_previews(output_dir, opt_config, scene_config, scene, integrator, 'final')
    print(f'[✔︎] Optimization complete: {opt_config.name}\n')
    print(f'Avg iteration time: {total_time / opt_config.n_iter:.4f}\n')

    if use_log:
        wandb.finish()

    return scene, params, opt
