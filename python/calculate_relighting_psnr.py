#!/usr/bin/env python3
"""
HDR EXR quality metrics: PSNR/SSIM/LPIPS variants for HDR image evaluation.
"""

import cv2
import numpy as np
import os
import math
import json
import torch
from skimage.metrics import structural_similarity as ssim

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# LPIPS model (lazy-loaded)
LPIPS_MODEL = None

def get_lpips_model():
    """Get or load the LPIPS model."""
    global LPIPS_MODEL
    if LPIPS_MODEL is None:
        try:
            import lpips
            LPIPS_MODEL = lpips.LPIPS(net='alex')  # use AlexNet as the backbone
            if torch.cuda.is_available():
                LPIPS_MODEL = LPIPS_MODEL.cuda()
            print("LPIPS model loaded successfully")
        except ImportError:
            print("Warning: the 'lpips' package is not installed; LPIPS will be unavailable")
            print("Install it with: pip install lpips")
            LPIPS_MODEL = False
        except Exception as e:
            print(f"Warning: failed to load LPIPS model: {str(e)}")
            LPIPS_MODEL = False
    return LPIPS_MODEL if LPIPS_MODEL is not False else None

def read_exr(file_path):
    """Read an EXR file."""
    img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Could not read file: {file_path}")
    return img

def calculate_psnr_linear(gt_image, render_image):
    """Standard PSNR (linear space)."""
    mse_r = np.mean((gt_image[:, :, 0] - render_image[:, :, 0]) ** 2)
    mse_g = np.mean((gt_image[:, :, 1] - render_image[:, :, 1]) ** 2)
    mse_b = np.mean((gt_image[:, :, 2] - render_image[:, :, 2]) ** 2)

    mse = (mse_r + mse_g + mse_b) / 3

    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(float(mse)))

    return float(psnr)

def calculate_psnr_log(gt_image, render_image, epsilon=1e-4):
    """Log-PSNR (logarithmic space) - better suited to HDR."""
    # Move to log space (add a small constant to avoid log(0))
    gt_log = np.log10(gt_image + epsilon)
    render_log = np.log10(render_image + epsilon)

    mse_r = np.mean((gt_log[:, :, 0] - render_log[:, :, 0]) ** 2)
    mse_g = np.mean((gt_log[:, :, 1] - render_log[:, :, 1]) ** 2)
    mse_b = np.mean((gt_log[:, :, 2] - render_log[:, :, 2]) ** 2)

    mse = (mse_r + mse_g + mse_b) / 3

    if mse == 0:
        return float('inf')

    # The log-space range is usually large; use a reasonable maximum here.
    # Assuming an HDR range of [epsilon, 100], log10 spans about [-4, 2], i.e. 6.
    max_log_value = 6.0
    psnr = 20 * math.log10(max_log_value / math.sqrt(float(mse)))

    return float(psnr)

def tonemapping_reinhard(hdr_image, key=0.18):
    """Reinhard tone mapping."""
    # Compute luminance
    luminance = 0.2126 * hdr_image[:, :, 0] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 2]

    # Avoid division by zero
    luminance = np.maximum(luminance, 1e-10)

    # Compute log-average luminance
    log_avg_lum = np.exp(np.mean(np.log(luminance + 1e-10)))

    # Scale luminance
    scaled_lum = (key / log_avg_lum) * luminance

    # Reinhard operator
    mapped_lum = scaled_lum / (1.0 + scaled_lum)

    # Apply to each channel
    scale = mapped_lum / (luminance + 1e-10)
    scale = scale[:, :, np.newaxis]

    ldr_image = hdr_image * scale
    ldr_image = np.clip(ldr_image, 0, 1)

    return ldr_image

def calculate_psnr_tonemapped(gt_image, render_image):
    """Tone-mapped PSNR (tone map first, then compute PSNR)."""
    gt_ldr = tonemapping_reinhard(gt_image)
    render_ldr = tonemapping_reinhard(render_image)

    return float(calculate_psnr_linear(gt_ldr, render_ldr))

def pu_encode(x):
    """Perceptually Uniform encoding."""
    # A PU21-like encoding, similar to the one used by HDR-VDP-2.
    # This is a simplified version.
    return np.where(x <= 0.0001,
                   0.0,
                   np.sign(x) * 10.0 * np.log10(1.0 + np.abs(x) * 100.0))

def calculate_psnr_pu(gt_image, render_image):
    """PU-PSNR (perceptually uniform space)."""
    # Convert to perceptually uniform space
    gt_pu = pu_encode(gt_image)
    render_pu = pu_encode(render_image)

    mse_r = np.mean((gt_pu[:, :, 0] - render_pu[:, :, 0]) ** 2)
    mse_g = np.mean((gt_pu[:, :, 1] - render_pu[:, :, 1]) ** 2)
    mse_b = np.mean((gt_pu[:, :, 2] - render_pu[:, :, 2]) ** 2)

    mse = (mse_r + mse_g + mse_b) / 3

    if mse == 0:
        return float('inf')

    # Reasonable range for PU space
    max_pu_value = 20.0
    psnr = 20 * math.log10(max_pu_value / math.sqrt(float(mse)))

    return float(psnr)

def calculate_relative_mse(gt_image, render_image, epsilon=1e-4):
    """Relative MSE - a common HDR metric."""
    diff = gt_image - render_image
    relative_error = diff / (gt_image + epsilon)
    relative_mse = np.mean(relative_error ** 2)
    return float(relative_mse)  # convert to a Python float

def calculate_ssim_hdr(gt_image, render_image, method='log'):
    """Compute SSIM for HDR images.

    Args:
        gt_image: ground-truth image
        render_image: rendered image
        method: 'linear', 'log', or 'tonemapped'
    """
    if method == 'log':
        # Log-space SSIM, better suited to HDR
        epsilon = 1e-4
        gt_log = np.log10(gt_image + epsilon)
        render_log = np.log10(render_image + epsilon)
        gt_proc = gt_log
        render_proc = render_log
    elif method == 'tonemapped':
        # SSIM after tone mapping
        gt_proc = tonemapping_reinhard(gt_image)
        render_proc = tonemapping_reinhard(render_image)
    else:  # linear
        gt_proc = gt_image
        render_proc = render_image

    # Compute SSIM per channel, then average
    ssim_values = []
    for i in range(3):  # BGR channels
        # data_range depends on the method
        if method == 'log':
            data_range = 6.0  # range of the log10 space
        elif method == 'tonemapped':
            data_range = 1.0
        else:
            data_range = float(gt_proc[:, :, i].max() - gt_proc[:, :, i].min())
            data_range = max(data_range, 1.0)  # avoid values that are too small

        ssim_val = ssim(gt_proc[:, :, i], render_proc[:, :, i],
                       data_range=data_range)
        ssim_values.append(float(ssim_val))

    return float(np.mean(ssim_values))

def calculate_lpips(gt_image, render_image, use_tonemapping=True):
    """Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        gt_image: ground-truth image
        render_image: rendered image
        use_tonemapping: whether to tone map first
    """
    lpips_model = get_lpips_model()
    if lpips_model is None:
        return None

    try:
        # Optionally tone map
        if use_tonemapping:
            gt_proc = tonemapping_reinhard(gt_image)
            render_proc = tonemapping_reinhard(render_image)
        else:
            # Just clip to [0, 1]
            gt_proc = np.clip(gt_image, 0, 1)
            render_proc = np.clip(render_image, 0, 1)

        # Convert to a torch tensor of shape [B, C, H, W] in range [-1, 1].
        # OpenCV reads BGR, so convert to RGB first.
        gt_rgb = cv2.cvtColor(gt_proc.astype(np.float32), cv2.COLOR_BGR2RGB)
        render_rgb = cv2.cvtColor(render_proc.astype(np.float32), cv2.COLOR_BGR2RGB)

        # Map [0, 1] to [-1, 1]
        gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        render_tensor = torch.from_numpy(render_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1

        if torch.cuda.is_available():
            gt_tensor = gt_tensor.cuda()
            render_tensor = render_tensor.cuda()

        # Compute LPIPS
        with torch.no_grad():
            lpips_value = lpips_model(gt_tensor, render_tensor)

        return float(lpips_value.item())

    except Exception as e:
        print(f"      LPIPS computation error: {str(e)}")
        return None

def calculate_all_metrics(gt_image, render_image):
    """Compute all metrics."""
    metrics = {}

    # PSNR - Linear
    try:
        metrics['psnr_linear'] = round(calculate_psnr_linear(gt_image, render_image), 4)
    except Exception as e:
        metrics['psnr_linear'] = None
        print(f"      Warning: Linear PSNR failed: {str(e)}")

    # PSNR - Log
    try:
        metrics['psnr_log'] = round(calculate_psnr_log(gt_image, render_image), 4)
    except Exception as e:
        metrics['psnr_log'] = None
        print(f"      Warning: Log PSNR failed: {str(e)}")

    # PSNR - Tonemapped
    try:
        metrics['psnr_tonemapped'] = round(calculate_psnr_tonemapped(gt_image, render_image), 4)
    except Exception as e:
        metrics['psnr_tonemapped'] = None
        print(f"      Warning: Tonemapped PSNR failed: {str(e)}")

    # PSNR - PU
    try:
        metrics['psnr_pu'] = round(calculate_psnr_pu(gt_image, render_image), 4)
    except Exception as e:
        metrics['psnr_pu'] = None
        print(f"      Warning: PU PSNR failed: {str(e)}")

    # Relative MSE
    try:
        metrics['relative_mse'] = round(calculate_relative_mse(gt_image, render_image), 6)
    except Exception as e:
        metrics['relative_mse'] = None
        print(f"      Warning: Relative MSE failed: {str(e)}")

    # SSIM - Linear
    try:
        metrics['ssim_linear'] = round(calculate_ssim_hdr(gt_image, render_image, method='linear'), 6)
    except Exception as e:
        metrics['ssim_linear'] = None
        print(f"      Warning: Linear SSIM failed: {str(e)}")

    # SSIM - Log (recommended for HDR)
    try:
        metrics['ssim_log'] = round(calculate_ssim_hdr(gt_image, render_image, method='log'), 6)
    except Exception as e:
        metrics['ssim_log'] = None
        print(f"      Warning: Log SSIM failed: {str(e)}")

    # SSIM - Tonemapped
    try:
        metrics['ssim_tonemapped'] = round(calculate_ssim_hdr(gt_image, render_image, method='tonemapped'), 6)
    except Exception as e:
        metrics['ssim_tonemapped'] = None
        print(f"      Warning: Tonemapped SSIM failed: {str(e)}")

    # LPIPS (with tone mapping)
    try:
        lpips_val = calculate_lpips(gt_image, render_image, use_tonemapping=True)
        if lpips_val is not None:
            metrics['lpips_tonemapped'] = round(lpips_val, 6)
        else:
            metrics['lpips_tonemapped'] = None
    except Exception as e:
        metrics['lpips_tonemapped'] = None
        print(f"      Warning: LPIPS failed: {str(e)}")

    return metrics

def process_scene(scene_path, scene_name, all_results):
    """Process a single scene."""
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")

    scene_results = {
        "scene": scene_name,
        "methods": {}
    }

    # Find all method folders
    method_folders = [d for d in os.listdir(scene_path)
                     if os.path.isdir(os.path.join(scene_path, d))]

    if not method_folders:
        print(f"Warning: no method folders found under scene {scene_name}")
        return

    print(f"Found {len(method_folders)} method folder(s)")

    # Iterate over each method folder
    for method_folder in method_folders:
        method_path = os.path.join(scene_path, method_folder)
        print(f"\nProcessing method: {method_folder}")

        # Find the gt folder
        gt_folder = os.path.join(method_path, 'gt')
        if not os.path.exists(gt_folder):
            print(f"  Warning: gt folder not found: {gt_folder}")
            continue

        # Read the gt image
        gt_exr_files = [f for f in os.listdir(gt_folder) if f.endswith('.exr')]
        if not gt_exr_files:
            print(f"  Warning: no EXR file in the gt folder")
            continue

        gt_exr_file = gt_exr_files[0]
        gt_path = os.path.join(gt_folder, gt_exr_file)

        try:
            gt_img = read_exr(gt_path)
            print(f"  Read GT: {gt_exr_file} (shape: {gt_img.shape})")
            print(f"  GT value range: [{gt_img.min():.4f}, {gt_img.max():.4f}]")
        except Exception as e:
            print(f"  Error: could not read GT image {gt_path}: {str(e)}")
            continue

        # Find all rendered-method subfolders
        render_folders = [d for d in os.listdir(method_path)
                         if os.path.isdir(os.path.join(method_path, d)) and d != 'gt']

        if not render_folders:
            print(f"  Warning: no render-result folders found")
            continue

        method_results = {}

        # Iterate over each rendered method
        for render_folder in render_folders:
            render_path = os.path.join(method_path, render_folder)
            print(f"    Processing render method: {render_folder}")

            # Read the rendered image
            render_exr_files = [f for f in os.listdir(render_path) if f.endswith('.exr')]
            if not render_exr_files:
                print(f"      Warning: no EXR file found")
                continue

            render_exr_file = render_exr_files[0]
            render_exr_path = os.path.join(render_path, render_exr_file)

            try:
                render_img = read_exr(render_exr_path)
                print(f"      Read render: {render_exr_file} (shape: {render_img.shape})")
                print(f"      Render value range: [{render_img.min():.4f}, {render_img.max():.4f}]")

                # Check that shapes match
                if gt_img.shape != render_img.shape:
                    print(f"      Warning: GT and render shapes differ ({gt_img.shape} vs {render_img.shape})")
                    continue

                # Compute all metrics
                metrics = calculate_all_metrics(gt_img, render_img)
                method_results[render_folder] = metrics

                print(f"      Linear PSNR: {metrics.get('psnr_linear', 'N/A')} dB")
                print(f"      Log PSNR: {metrics.get('psnr_log', 'N/A')} dB")
                print(f"      Tonemapped PSNR: {metrics.get('psnr_tonemapped', 'N/A')} dB")
                print(f"      PU PSNR: {metrics.get('psnr_pu', 'N/A')} dB")
                print(f"      Relative MSE: {metrics.get('relative_mse', 'N/A')}")
                print(f"      Linear SSIM: {metrics.get('ssim_linear', 'N/A')}")
                print(f"      Log SSIM: {metrics.get('ssim_log', 'N/A')}")
                print(f"      Tonemapped SSIM: {metrics.get('ssim_tonemapped', 'N/A')}")
                print(f"      LPIPS (Tonemapped): {metrics.get('lpips_tonemapped', 'N/A')}")

            except Exception as e:
                print(f"      Error: failed while processing render image {render_exr_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if method_results:
            scene_results["methods"][method_folder] = method_results

    if scene_results["methods"]:
        all_results[scene_name] = scene_results

def main():
    """Entry point."""
    import argparse
    parser = argparse.ArgumentParser('calculate_relighting_psnr.py')
    parser.add_argument('base_path',
                        help='directory containing the per-scene "<scene>-render" relighting outputs')
    args = parser.parse_args()
    base_path = args.base_path
    output_json = os.path.join(base_path, "hdr_psnr_results.json")

    print("=" * 80)
    print("HDR EXR quality-metrics script")
    print("=" * 80)
    print(f"Base path: {base_path}")
    print(f"Output JSON: {output_json}")
    print("\nSupported metrics:")
    print("  1. PSNR (Linear)      - traditional PSNR in linear space")
    print("  2. PSNR (Log)         - PSNR in log space, suited to HDR")
    print("  3. PSNR (Tonemapped)  - PSNR after tone mapping")
    print("  4. PSNR (PU)          - PSNR in perceptually uniform space")
    print("  5. Relative MSE       - relative mean squared error")
    print("  6. SSIM (Linear)      - SSIM in linear space")
    print("  7. SSIM (Log)         - SSIM in log space, suited to HDR")
    print("  8. SSIM (Tonemapped)  - SSIM after tone mapping")
    print("  9. LPIPS (Tonemapped) - perceptual similarity (lower is better)")

    # Check that the base path exists
    if not os.path.exists(base_path):
        print(f"Error: base path does not exist: {base_path}")
        return

    # Gather all scene folders
    scene_folders = [d for d in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, d)) and d.endswith('-render')]

    print(f"\nFound {len(scene_folders)} scene(s):")
    for scene in scene_folders:
        print(f"  - {scene}")

    # Store all results
    all_results = {
        "metadata": {
            "description": "HDR image quality metrics",
            "metrics": {
                "psnr_linear": "Traditional PSNR in linear space",
                "psnr_log": "PSNR in logarithmic space (better for HDR)",
                "psnr_tonemapped": "PSNR after Reinhard tone mapping",
                "psnr_pu": "PSNR in perceptually uniform space",
                "relative_mse": "Relative mean squared error",
                "ssim_linear": "SSIM in linear space",
                "ssim_log": "SSIM in logarithmic space (better for HDR)",
                "ssim_tonemapped": "SSIM after tone mapping",
                "lpips_tonemapped": "LPIPS (perceptual similarity) after tone mapping (lower is better)"
            }
        },
        "scenes": {}
    }

    # Process each scene
    for scene_folder in sorted(scene_folders):
        scene_path = os.path.join(base_path, scene_folder)
        scene_name = scene_folder.replace('-render', '')

        try:
            process_scene(scene_path, scene_name, all_results["scenes"])
        except Exception as e:
            print(f"\nError: failed while processing scene {scene_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save results to JSON
    if all_results["scenes"]:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print(f"Done! Processed {len(all_results['scenes'])} scene(s)")
        print(f"Results written to: {output_json}")
        print("=" * 80)

        # Print a summary
        print("\nSummary:")
        for scene_name, scene_data in all_results["scenes"].items():
            print(f"\nScene: {scene_name}")
            for method, renders in scene_data["methods"].items():
                print(f"  Method: {method}")
                for render_name, metrics in renders.items():
                    print(f"    {render_name}:")
                    for metric_name, value in metrics.items():
                        if value is not None:
                            print(f"      {metric_name}: {value}")
    else:
        print("\nWarning: no scenes were processed successfully")

if __name__ == "__main__":
    main()
