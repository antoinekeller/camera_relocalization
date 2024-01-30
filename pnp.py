"""
Core algorithm to perform the camera pose estimation on a video.

Once the CNN is trained, you will be able to export the detected keypoints
on the video. For developpement reasons, it was easier to run the inference separately
and to store the results in a json (with confidence). At some point, it will be cleaner
to have the inference called on the fly.

Pass the PLY, the intrinsics camera matrix, the keypoints detected, the 3D world
coordinates of the keypoints, and run openCV PNP.

Point cloud will be reprojected to give you a feeling of the expected accuracy
"""
import json
from time import time
from argparse import ArgumentParser
import numpy as np
import cv2
from plyfile import PlyData
import matplotlib.pyplot as plt
from read_write_model import read_cameras_binary

# Threshold to decide if a keypoints must be taken into account in the PnP
CONFIDENCE_THRESHOLD = 0.05


def write_image_status(img, ok_reloc):
    """Display OK/Lost at the bottom left"""
    cv2.putText(
        img,
        "OK" if ok_reloc else "Lost",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(0, 255, 0) if ok_reloc else (0, 0, 255),
        thickness=3,
    )


def read_intrinsics():
    """
    Read intrinsics from COLMAP results
    We use the OPENCV camera model (with distortion)
    """
    cameras = read_cameras_binary(f"{args.colmap}/cameras.bin")

    K = np.identity(3)
    K[0, 0] = cameras[1].params[0]
    K[1, 1] = cameras[1].params[1]
    K[0, 2] = cameras[1].params[2]
    K[1, 2] = cameras[1].params[3]

    print("intr", cameras[1].params)

    dist_coeffs = cameras[1].params[4:8]
    print(dist_coeffs)

    print(K)

    return K, dist_coeffs


def draw_ply_points(img, ply_pixels, my_depths):
    """
    This function is a bit tricky. The goal is to display the projected point cloud
    once the camera pose is found. As you have many points, using the OpenCV classical
    functions will be too slow, so I hacked a bit and used numpy to display crosses.
    Depth is also taken into account with a nice depthmap correspondance.

    Write a simple opencv method calling cv2.circle() if you find this too cumbersome
    """
    flat_index_array = np.ravel_multi_index(ply_pixels, (img.shape[0], img.shape[1]))

    # Draw little cross
    index_array = np.concatenate(
        [
            flat_index_array,
            flat_index_array - 1,
            flat_index_array + 1,
            flat_index_array - img.shape[1],
            flat_index_array + img.shape[1],
        ]
    )

    unique_index_array, idx_start = np.unique(index_array, return_index=True)

    depth = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    np.ravel(depth)[unique_index_array] = True

    my_5depths = np.concatenate(
        [
            my_depths,
            my_depths,
            my_depths,
            my_depths,
            my_depths,
        ]
    )

    depth_min = np.min(my_depths)
    depth_max = np.max(my_depths)
    depth_factor = depth_max - depth_min

    colors = colormap((depth_max - my_5depths[idx_start]) / depth_factor)[:, :3] * 255

    img[depth] = colors


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("colmap", help="Colmap folder")
    parser.add_argument("video", help="Video to be tested")
    parser.add_argument("keypoints", help="Keypoints infered with NN module")
    args = parser.parse_args()

    K, dist_coeffs = read_intrinsics()

    colormap = plt.get_cmap("viridis")

    points_in_world = np.loadtxt(f"{args.colmap}/points.txt").reshape(-1, 3)

    print(f"Load {points_in_world.shape[0]} points in world")

    with open(args.keypoints) as f:
        keypoints = json.load(f)

    with open(f"{args.colmap}/cleaned.ply", "rb") as f:
        plydata = PlyData.read(f)
    nb_points_ply = plydata.elements[0].count
    print(f"Loaded {nb_points_ply} points from {args.colmap}/cleaned.ply")

    ply_points = np.zeros((nb_points_ply, 3))

    ply_points[:, 0] = plydata.elements[0].data["x"]
    ply_points[:, 1] = plydata.elements[0].data["y"]
    ply_points[:, 2] = plydata.elements[0].data["z"]

    useExtrinsicGuess = False

    # cap = cv2.VideoCapture(f"{args.colmap}/video.mp4")
    cap = cv2.VideoCapture(args.video)

    out = cv2.VideoWriter(
        "out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 1080)
    )

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video stream or file")

    for idx in range(n_frames):
        # Capture frame-by-frame
        ret, img = cap.read()
        if not ret:
            break

        points_in_pixel = np.array(keypoints[f"frame_{idx+1:03d}.png"])
        mask_prob = (
            points_in_pixel[:, 2] > CONFIDENCE_THRESHOLD
        )  # filter out low confidence points
        points_in_pixel = points_in_pixel[:, :2]
        points_in_pixel = points_in_pixel.astype(float)

        for i, keypoint in enumerate(points_in_pixel):
            cv2.circle(
                img,
                center=(int(keypoint[0]), int(keypoint[1])),
                radius=3,
                color=(0, 255, 0) if mask_prob[i] else (0, 0, 255),
                thickness=2,
            )

        if np.count_nonzero(mask_prob) < 6:
            print("Not enough points")
            write_image_status(img, ok_reloc=False)

            out.write(img)
            cv2.imshow("Image", img)
            k = cv2.waitKey(1)

            if k == 27:
                break

            useExtrinsicGuess = False
            continue

        if not useExtrinsicGuess:
            guess_rot = np.zeros((1, 3), dtype=float)
            guess_trans = (0, 0, 0)

        # Perspective-n-Point algorithm, returning rotation and translation vector
        ret, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            points_in_world[mask_prob],
            points_in_pixel[mask_prob],
            K,
            distCoeffs=dist_coeffs,
            rvec=guess_rot,
            tvec=guess_trans,
            useExtrinsicGuess=useExtrinsicGuess,
        )

        if ret:
            useExtrinsicGuess = True
            guess_rot = rotation_vector
            guess_trans = translation_vector
        else:
            print("Ret", ret)
            write_image_status(img, ok_reloc=False)
            out.write(img)
            cv2.imshow("Image", img)
            k = cv2.waitKey(1)

            if k == 27:
                break

            useExtrinsicGuess = False
            continue

        # in the reference world
        to_device_from_world_rot = cv2.Rodrigues(rotation_vector)[0]

        start = time()

        ply_in_cam = to_device_from_world_rot.dot(ply_points.T) + translation_vector

        my_depths = ply_in_cam[2, :].copy()

        ply_pixels = cv2.projectPoints(
            ply_in_cam.T, np.zeros(3), np.zeros(3), K, dist_coeffs
        )

        ply_pixels = ply_pixels[0].astype(int).reshape(-1, 2)

        # print("computation", time() - start)
        start = time()
        ply_pixels = ply_pixels[:, [1, 0]]

        mask_oob = (
            (ply_pixels[:, 0] >= 1)
            & (ply_pixels[:, 0] < img.shape[0] - 1)
            & (ply_pixels[:, 1] >= 1)
            & (ply_pixels[:, 1] < img.shape[1] - 1)
        )

        ply_pixels = ply_pixels[mask_oob].T
        my_depths = my_depths[mask_oob]

        assert len(my_depths) > 0

        draw_ply_points(img, ply_pixels, my_depths)

        # print(depth)
        write_image_status(img, ok_reloc=True)
        out.write(img)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)

        if k == 27:
            break

    out.release()
