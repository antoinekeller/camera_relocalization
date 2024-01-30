"""
Project 3D Keypoints to each images used in COLMAP. For this, use the intrinsics
(cameras.bin), extrinsics poses (images.bin) and points in world described in a text file

Open3D is also used to build a voxel grid of the point cloud, so that we can roughly
estimate if a point is hidden by the structure. Parameters (like voxel size) must be adjusted
accordingly.

In the end, everything is dumped to a json file that will feed our NN.
"""
from argparse import ArgumentParser
import json
import numpy as np
import cv2
import open3d as o3d
from read_write_model import read_cameras_binary, read_images_binary, qvec2rotmat


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("colmap", help="Colmap folder")
    parser.add_argument("keypoints_in_world", help="Keypoints in world txt file")
    args = parser.parse_args()

    # Careful: poses are not ordered !
    poses = read_images_binary(f"{args.colmap}/images.bin")
    cameras = read_cameras_binary(f"{args.colmap}/cameras.bin")

    # Sort poses by frame name
    poses = sorted(poses.values(), key=lambda pose: pose.name)

    pcd = o3d.io.read_point_cloud(f"{args.colmap}/cleaned.ply")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)

    # o3d.visualization.draw_geometries([voxel_grid])

    K = np.identity(3)
    K[0, 0] = cameras[1].params[0]
    K[1, 1] = cameras[1].params[1]
    K[0, 2] = cameras[1].params[2]
    K[1, 2] = cameras[1].params[3]

    dist_coeffs = cameras[1].params[4:8]
    print(dist_coeffs)

    print("intr", cameras[1].params)

    print(K)

    points_in_world = np.loadtxt(args.keypoints_in_world).reshape(-1, 3)

    nb_keypoints = len(points_in_world)

    keypoints = {}

    for pose in poses:
        img = cv2.imread(f"{args.colmap}/images/{pose.name}")

        points_in_cam = qvec2rotmat(pose.qvec).dot(
            points_in_world.T
        ) + pose.tvec.reshape(3, 1)

        cam_position = (
            -qvec2rotmat(pose.qvec).T.dot(pose.tvec.reshape(3, 1))
        ).flatten()

        visible = np.ones(nb_keypoints, dtype=bool)

        xy_projected = cv2.projectPoints(
            points_in_cam.T, np.zeros(3), np.zeros(3), K, dist_coeffs
        )

        points_in_pixel = xy_projected[0].reshape((nb_keypoints, 2))

        for i, keypoint in enumerate(points_in_pixel):
            visible[i] = (0 <= keypoint[0] < img.shape[1]) & (
                0 <= keypoint[1] < img.shape[0]
            )

        for i in range(nb_keypoints):
            if not visible[i]:
                continue

            d = points_in_world[i] - cam_position
            dist = np.linalg.norm(d)
            d /= dist
            arange = np.arange(dist, 0, -0.05).reshape(-1, 1)
            points_to_check = cam_position + d * arange

            voxel_checks = voxel_grid.check_if_included(
                o3d.utility.Vector3dVector(points_to_check)
            )

            if len(np.argwhere(voxel_checks)) == 0:
                visible[i] = True
            elif np.argwhere(voxel_checks).flatten()[-1] <= 5:
                visible[i] = True
            else:
                visible[i] = False

        points = np.hstack((points_in_pixel, visible.reshape(-1, 1)))

        frame = {pose.name: points.tolist()}

        for i, keypoint in enumerate(points_in_pixel):
            cv2.circle(
                img,
                center=(int(keypoint[0]), int(keypoint[1])),
                radius=3,
                color=(255, 255, 0) if visible[i] else (0, 0, 255),
                thickness=2,
            )

        keypoints.update(frame)

        cv2.imshow("Image", img)
        k = cv2.waitKey(0)

        if k == 27:
            break

    with open("keypoints.json", "w") as f:
        json.dump(keypoints, f)
