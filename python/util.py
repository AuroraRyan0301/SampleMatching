"""
Various helper functions.
"""
import os
import math
import atexit
import threading
import subprocess

import numpy as np


def calculate_psnr(gt_image, render_image, max_value=1.0):
    """PSNR (dB) between two HDR/LDR images given as (H, W, C) numpy arrays.

    Channels are averaged. Returns inf when the images are identical.
    """
    mse = np.mean((gt_image[..., :3] - render_image[..., :3]) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_value / math.sqrt(mse))


# ---- Async zstd worker pool (used by save_params when async_compression=True)
_ZSTD_MAX_CONCURRENT = 4
_zstd_pending = []  # list[Thread]
_zstd_lock = threading.Lock()


def _zstd_worker(payload, actual_path):
    """Run zstd compression on a single payload. Called from a worker thread."""
    proc = subprocess.Popen(
        ['zstd', '-q', '-3', '-f', '-o', actual_path],
        stdin=subprocess.PIPE)
    proc.communicate(input=payload)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)


def _zstd_submit_async(payload, actual_path):
    """Submit a zstd job. Blocks if too many concurrent."""
    with _zstd_lock:
        # Reap completed threads
        _zstd_pending[:] = [t for t in _zstd_pending if t.is_alive()]
        # If at max, wait for the oldest to finish
        while len(_zstd_pending) >= _ZSTD_MAX_CONCURRENT:
            t = _zstd_pending.pop(0)
            t.join()
            _zstd_pending[:] = [t for t in _zstd_pending if t.is_alive()]
        t = threading.Thread(target=_zstd_worker,
                             args=(payload, actual_path),
                             daemon=False)
        t.start()
        _zstd_pending.append(t)


def _zstd_drain_all():
    """Wait for all pending zstd workers to finish."""
    with _zstd_lock:
        threads = list(_zstd_pending)
        _zstd_pending.clear()
    for t in threads:
        t.join()


atexit.register(_zstd_drain_all)


def _free_gb(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / (1024 ** 3)


def linear_to_srgb(img):
    """Linear → sRGB tonemap with hard clamp at 1.0."""
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * np.power(np.clip(img, 0, None), 1 / 2.4) - 0.055,
                   12.92 * img)
    return np.clip(img, 0, 1)


def write_preview_png(image, png_path, free_threshold_gb=0.0):
    """Tonemap a linear HDR mi.Image / Bitmap-like to sRGB and write as PNG.
    If the destination partition has < free_threshold_gb free, skip silently.
    Returns True if written, False if skipped.
    """
    parent = os.path.dirname(png_path)
    os.makedirs(parent, exist_ok=True)
    if free_threshold_gb > 0 and _free_gb(parent) < free_threshold_gb:
        return False
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    srgb = linear_to_srgb(arr)
    u8 = (srgb * 255 + 0.5).astype(np.uint8)
    import cv2
    cv2.imwrite(png_path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    return True


def _vol_bytes(arr):
    """Serialize a numpy array to Mitsuba .vol format in memory.

    Mitsuba .vol layout: 'VOL' + version=3 + encoding=1 (float32)
    + (X, Y, Z, channels) + bbox (6 floats) + raw float32 data.

    Numpy axis order is (Z, Y, X[, C]); header writes (X, Y, Z, C).
    bbox is hardcoded to the unit cube (matching mi.VolumeGrid(arr).write).
    """
    import struct
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        Z, Y, X = arr.shape; C = 1
    elif arr.ndim == 4:
        Z, Y, X, C = arr.shape
    else:
        raise ValueError(f'Unexpected vol array ndim={arr.ndim}, shape={arr.shape}')
    header = struct.pack('<3sBi4i6f', b'VOL', 3, 1, X, Y, Z, C,
                         0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    return header + arr.tobytes()


def save_params(output_dir, scene_config, params, name,
                skip_emission=False, compression=None,
                async_compression=False):
    """Save volume params for the given checkpoint name.

    skip_emission         : drop any param key containing 'emission'.
    compression           : None or 'zstd' (zstd-3 via stdin pipe).
    async_compression     : if True (and compression='zstd'), zstd runs in a
                            background thread.  An atexit hook drains pending
                            workers before the process exits.
    """
    os.makedirs(output_dir, exist_ok=True)
    for key in scene_config.param_keys:
        if skip_emission and 'emission' in key:
            continue
        value = params[key]
        if not key.endswith('.data'):
            raise NotImplementedError(f'Checkpointing of parameter {key} with type {type(value)}')

        for suffix in ['.data', '.values', '.value']:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
        var_name = '_'.join(key.strip().split('.'))

        target_vol = os.path.join(output_dir, f'{name}-{var_name}.vol')

        # Serialize in-memory (no temp file, no orphan on failure).
        payload = _vol_bytes(value.numpy())

        if compression == 'zstd':
            target_path = target_vol + '.zst'
            if async_compression:
                _zstd_submit_async(payload, target_path)
            else:
                # synchronous: pipe payload to zstd
                subprocess.run(['zstd', '-q', '-3', '-f', '-o', target_path],
                               input=payload, check=True)
        else:
            with open(target_vol, 'wb') as f:
                f.write(payload)



def get_single_medium(scene):
    """
    Since we only support a very restricted setup (single medium within a single
    bounding shape), we can extract the only medium pointer within the scene
    and use is for all subsequent method calls. This avoids expensive virtual
    function calls on array pointers.
    """
    shapes = scene.shapes()
    assert len(shapes) == 1, f'Not supported: more than 1 shape in the scene (found {len(shapes)}).'
    medium = shapes[0].interior_medium()
    assert medium is not None, 'Expected a single shape with an interior medium.'
    return medium
