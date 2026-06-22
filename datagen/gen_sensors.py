import numpy as np

# ------------------------------------------------------------
# 1. Uniform sampling on a sphere (Marsaglia method)
# ------------------------------------------------------------
def sample_points_on_sphere(N):
    pts = []
    for _ in range(N):
        while True:
            x1, x2 = np.random.uniform(-1, 1, 2)
            s = x1*x1 + x2*x2
            if s >= 1:
                continue
            x = 2 * x1 * np.sqrt(1 - s)
            y = 2 * x2 * np.sqrt(1 - s)
            z = 1 - 2 * s
            pts.append(np.array([x, y, z], dtype=float))
            break
    return np.array(pts)


# ------------------------------------------------------------
# 2. Generate sensor origins around a bounding box
# ------------------------------------------------------------
def generate_sensor_positions(bbox_min, bbox_max, N=32, distance_factor=2.5):
    bbox_min = np.array(bbox_min, dtype=float)
    bbox_max = np.array(bbox_max, dtype=float)

    center = 0.5 * (bbox_min + bbox_max)
    diag   = (bbox_max - bbox_min)
    radius = 0.5 * np.linalg.norm(diag)

    distance = distance_factor * radius

    dirs = sample_points_on_sphere(N)
    origins = center + dirs * distance

    return origins, center, dirs


# ------------------------------------------------------------
# 3. Choose a stable up vector
# ------------------------------------------------------------
def up_vector(direction):
    # If camera direction is near +Y or -Y, use X-axis as up vector
    if abs(direction[1]) > 0.9:
        return np.array([1, 0, 0], dtype=float)
    return np.array([-0.31, 0.31, 0.89], dtype=float)


# ------------------------------------------------------------
# 4. Write all sensors into a .xml file
# ------------------------------------------------------------
def write_sensors_to_xml(filename, origins, center, dirs,
                         width=512, height=512, fov=45.0):
    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<scene version="2.0.0">\n\n')

        for i, org in enumerate(origins):
            up = up_vector(dirs[i])

            f.write(f'    <!-- Sensor {i} -->\n')
            f.write('    <sensor type="perspective">\n')
            f.write(f'        <float name="fov" value="{fov}"/>\n')
            f.write('        <transform name="to_world">\n')
            f.write(f'            <lookat origin="{org[0]} {org[1]} {org[2]}" '
                    f'target="{center[0]} {center[1]} {center[2]}" '
                    f'up="{up[0]} {up[1]} {up[2]}"/>\n')
            f.write('        </transform>\n')
            f.write('        <film type="hdrfilm">\n')
            f.write(f'            <integer name="width" value="{width}"/>\n')
            f.write(f'            <integer name="height" value="{height}"/>\n')
            f.write('        </film>\n')
            f.write('    </sensor>\n\n')

        f.write('</scene>\n')


# ------------------------------------------------------------
# 5. Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    bbox_min = (-1.33877, -1.76184, -0.0457303)
    bbox_max = (1.34527, 2.27303, 1.76553)

    origins, center, dirs = generate_sensor_positions(
        bbox_min, bbox_max,
        N=65,
        distance_factor=2.0
    )

    write_sensors_to_xml("sensors.xml", origins, center, dirs,
                         width=683, height=512, fov=55.0)

    print("✔ sensors.xml created!")