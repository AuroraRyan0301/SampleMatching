import numpy as np
from noise import pnoise3
import struct

# ----------------------------
# Mitsuba .vol writers
# ----------------------------

def write_mitsuba_vol_scalar(filename, grid, bbox=(0, 0, 0, 1, 1, 1)):
    """
    Write a scalar 3D grid to Mitsuba .vol format (version 3, float32).
    grid: (nx, ny, nz)
    """
    grid = np.asarray(grid, dtype=np.float32)
    assert grid.ndim == 3
    nx, ny, nz = grid.shape
    channels = 1

    bbox = np.asarray(bbox, dtype=np.float32)
    assert bbox.shape == (6,)

    with open(filename, "wb") as f:
        # Magic
        f.write(b"VOL")

        # Version (byte)
        f.write(struct.pack("b", 3))

        # Encoding (int32) -> 1 = float32
        f.write(struct.pack("<i", 1))

        # Dimensions (int32)
        f.write(struct.pack("<i", nx))
        f.write(struct.pack("<i", ny))
        f.write(struct.pack("<i", nz))

        # Channels (int32)
        f.write(struct.pack("<i", channels))

        # Bounding box
        f.write(bbox.astype(np.float32).tobytes())

        # Data: x fastest, then y, then z
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    f.write(struct.pack("<f", float(grid[x, y, z])))


def write_mitsuba_vol_rgb(filename, grid_rgb, bbox=(0, 0, 0, 1, 1, 1)):
    """
    Write an RGB 3D grid to Mitsuba .vol format (version 3, float32).
    grid_rgb: (nx, ny, nz, 3)
    """
    grid_rgb = np.asarray(grid_rgb, dtype=np.float32)
    assert grid_rgb.ndim == 4 and grid_rgb.shape[3] == 3
    nx, ny, nz, ch = grid_rgb.shape
    channels = 3

    bbox = np.asarray(bbox, dtype=np.float32)
    assert bbox.shape == (6,)

    with open(filename, "wb") as f:
        # Magic
        f.write(b"VOL")

        # Version (byte)
        f.write(struct.pack("b", 3))

        # Encoding (int32) -> 1 = float32
        f.write(struct.pack("<i", 1))

        # Dimensions
        f.write(struct.pack("<i", nx))
        f.write(struct.pack("<i", ny))
        f.write(struct.pack("<i", nz))

        # Channels
        f.write(struct.pack("<i", channels))

        # Bounding box
        f.write(bbox.astype(np.float32).tobytes())

        # Data order: ((z * yres + y) * xres + x) * channels + c
        # i.e. x fastest, then y, then z, then channel
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    rgb = grid_rgb[x, y, z]
                    for c in range(3):
                        f.write(struct.pack("<f", float(rgb[c])))

# ----------------------------
# Medium generation
# ----------------------------

def fbm_noise3(x, y, z, octaves=4, persistence=0.5, lacunarity=2.0, base=0):
    """Simple FBM wrapper around pnoise3, result approx in [-1,1]."""
    value = 0.0
    amp = 1.0
    freq = 1.6
    max_amp = 0.0
    for _ in range(octaves):
        value += amp * pnoise3(x * freq, y * freq, z * freq,
                               repeatx=1024, repeaty=1024, repeatz=1024,
                               base=base)
        max_amp += amp
        amp *= persistence
        freq *= lacunarity
    return value / max_amp  # normalize-ish to [-1,1]


def generate_sigma_t_and_colored_albedo(
    nx=64, ny=64, nz=64,
    scale=0.05,
    sigma_t_min=0.0,
    sigma_t_max=2.0,
    albedo_min=0.3,
    albedo_max=0.98,
    base_color=np.array([0.7, 0.4, 0.2], dtype=np.float32),
    density_exponent=1.3,
):
    """
    Generate scalar sigma_t and RGB albedo where R,G,B are different.
    Returns:
        sigma_t: (nx, ny, nz)
        albedo_rgb: (nx, ny, nz, 3)
    """
    sigma_t = np.zeros((nx, ny, nz), dtype=np.float32)
    albedo_rgb = np.zeros((nx, ny, nz, 3), dtype=np.float32)

    # Offsets for density + per-channel color noise
    offset_density = np.random.rand(3) * 100.0
    offset_r = np.random.rand(3) * 100.0 + 200.0
    offset_g = np.random.rand(3) * 100.0 + 400.0
    offset_b = np.random.rand(3) * 100.0 + 600.0

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = ix * scale
                y = iy * scale
                z = iz * scale

                # --- density / sigma_t ---
                nd = fbm_noise3(x + offset_density[0],
                                y + offset_density[1],
                                z + offset_density[2],
                                base=0)
                nd01 = 0.5 * (nd + 1.0)       # [0,1]
                density = nd01**density_exponent
                sigma_t_val = sigma_t_min + density * (sigma_t_max - sigma_t_min)
                sigma_t[ix, iy, iz] = sigma_t_val

                # --- color tint: 3 independent FBM fields ---
                nr = fbm_noise3(x + offset_r[0],
                                y + offset_r[1],
                                z + offset_r[2],
                                base=1)
                ng = fbm_noise3(x + offset_g[0],
                                y + offset_g[1],
                                z + offset_g[2],
                                base=2)
                nb = fbm_noise3(x + offset_b[0],
                                y + offset_b[1],
                                z + offset_b[2],
                                base=3)

                r = 0.5 * (nr + 1.0)
                g = 0.5 * (ng + 1.0)
                b = 0.5 * (nb + 1.0)

                # Slightly bias color to avoid all-gray look
                color = np.array([r, g, b], dtype=np.float32)
                color = np.clip(color, 0.0, 1.0)

                # Base albedo brightness depending on density
                base_albedo = albedo_min + (1.0 - density) * (albedo_max - albedo_min)

                # Final RGB albedo = base brightness * color tint
                a_rgb = base_albedo * base_color
                a_rgb = np.clip(a_rgb, 0.0, 1.0)

                albedo_rgb[ix, iy, iz, :] = a_rgb

    return sigma_t, albedo_rgb


if __name__ == "__main__":
    nx = 128
    ny = 256
    nz = 128

    sigma_t, albedo_rgb = generate_sigma_t_and_colored_albedo(
        nx=nx, ny=ny, nz=nz,
        scale=0.2,
        sigma_t_min=5.0,
        sigma_t_max=11.0,
        albedo_min=0.5,
        albedo_max=2.5,
        base_color=np.array([53.0/255, 94.0/255, 59.0/255], dtype=np.float32),
    )

    # Write volumes for Mitsuba
    bbox = (-1.8, -2.2, -0.5, 1.8, 2.8, 2.5)
    write_mitsuba_vol_scalar("sigma_t.vol", sigma_t, bbox=bbox)
    write_mitsuba_vol_rgb("albedo_rgb.vol", albedo_rgb, bbox=bbox)
