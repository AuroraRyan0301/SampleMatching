import numpy as np

def spherical_fibonacci_points(n):
    """
    Generate n points uniformly distributed on a sphere
    using the spherical Fibonacci point set.
    """
    i = np.arange(0, n)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * i / (n - 1)     # evenly spaced z values
    radius_xy = np.sqrt(1 - z * z)

    x = radius_xy * np.cos(theta)
    y = radius_xy * np.sin(theta)

    return np.vstack([x, y, z]).T


def generate_sensors_xml(target, radius, sensor_count, fov=45):
    """
    Generate a complete Mitsuba XML scene containing multiple sensors.
    Only wraps your existing sensor blocks with <scene>...</scene>.
    """
    tx, ty, tz = target
    pts = spherical_fibonacci_points(sensor_count)

    sensor_blocks = []
    for idx, p in enumerate(pts):
        ox, oy, oz = p * radius + np.array(target)

        block = f"""
    <sensor type="perspective">
        <float name="fov" value="{fov:.6f}"/>
        <transform name="to_world">
            <lookat origin="{ox:.6f}, {oy:.6f}, {oz:.6f}"
                    target="{tx:.6f}, {ty:.6f}, {tz:.6f}"
                    up    ="0.00, 1.00, 0.00"/>
        </transform>
        <ref id="sampler"/>
        <ref id="film"/>
    </sensor>"""

        sensor_blocks.append(block)

    sensors_xml = "\n".join(sensor_blocks)

    # --- wrap with <scene> ---
    full_xml = f"""<scene version="2.0.0">
{sensors_xml}
</scene>
"""

    return full_xml


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('sensor_gen.py')
    parser.add_argument('output', help='output sensors.xml path')
    parser.add_argument('--radius', type=float, default=3.6)
    parser.add_argument('--sensor-count', type=int, default=64)
    parser.add_argument('--fov', type=float, default=30)
    parser.add_argument('--target', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Fix random seed for reproducibility
    np.random.seed(args.seed)

    xml = generate_sensors_xml(
        target=tuple(args.target),
        radius=args.radius,
        sensor_count=args.sensor_count,
        fov=args.fov,
    )

    with open(args.output, "w") as f:
        f.write(xml)

    print(f"Generated {args.sensor_count} sensors and saved to {args.output}")
