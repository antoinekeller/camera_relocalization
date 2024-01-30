"""
With this script you can convert a points3D.bin representing point cloud
generated from COLMAP to a PLY file.
"""

from argparse import ArgumentParser
import numpy as np
from plyfile import PlyElement, PlyData
from read_write_model import read_points3D_binary


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("points3D_bin")
    parser.add_argument("ply")
    args = parser.parse_args()

    points = read_points3D_binary(args.points3D_bin)

    nb_points = len(points.keys())

    print(nb_points)

    xyz = np.zeros((nb_points, 3), dtype=np.float32)
    rgb = np.zeros((nb_points, 3), dtype=np.uint8)

    for i, point in enumerate(points.values()):
        xyz[i] = point.xyz
        rgb[i] = point.rgb

    pts = list(zip(xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb[:, 0], rgb[:, 1], rgb[:, 2]))

    vertex = np.array(
        pts,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(args.ply)
