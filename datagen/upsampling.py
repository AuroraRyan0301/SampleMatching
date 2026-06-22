import mitsuba as mi
mi.set_variant("cuda_ad_rgb")   # or your own variant

import numpy as np
import scipy.ndimage as ndi

def load_grid_np(fn):
    g = mi.load_dict({'type': 'gridvolume', 'filename': fn})
    arr = np.array(mi.traverse(g)['data'].numpy())
    return arr

def upsample_vol(input_path, output_path, target_res):
    """
    input_path  : input .vol file
    output_path : output .vol file
    target_res  : target resolution (nx, ny, nz)
    """

    grid = load_grid_np(input_path)
    # import ipdb; ipdb.set_trace()

    orig_res = grid.shape[:3]
    channels = grid.shape[3]

    print(f"Original resolution = {orig_res}, channels = {channels}")
    print(f"Target resolution   = {target_res}")

    # 2. compute zoom factor
    zoom_factor = (
        target_res[0] / orig_res[0],
        target_res[1] / orig_res[1],
        target_res[2] / orig_res[2],
        1   # do not scale channels
    )

    # 3. upsample
    up_grid = ndi.zoom(grid, zoom=zoom_factor, order=1)

    print("Upsampled grid shape =", up_grid.shape)

    # 4. save as a new .vol
    new_vol = mi.VolumeGrid(up_grid)
    new_vol.write(output_path)

    print("Saved to:", output_path)


# --------------------------------------------------------
#  Example: python upsampling.py in.vol out.vol 320 320 320
# --------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('upsampling.py')
    parser.add_argument('input', help='input .vol file')
    parser.add_argument('output', help='output .vol file')
    parser.add_argument('res', type=int, nargs=3, metavar=('NX', 'NY', 'NZ'),
                        help='target resolution')
    args = parser.parse_args()
    upsample_vol(args.input, args.output, target_res=tuple(args.res))