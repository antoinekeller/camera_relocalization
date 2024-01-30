# Efficient camera relocalization with keypoints detection and PnP

The goal of this project is to be able to recover camera pose from a single image with respect to the 3D point cloud of a scene. First we use COLMAP or any SfM software to reconstruct a 3D mapping of the object (and simultaneously the poses) from a video. Then you can (manually) select 3D keypoints in a well-distributed way, and a CNN can be trained to find the positions of these keypoints projected to every images. In the end, after detecting the keypoints, you can recover the pose with the Perspective-n-Point algorithm.

[YouTube video here](https://youtu.be/s7Q6OW3AXKU)

<p>
<em>Relocalization</em></br>
<img src="res/reloc.gif"  width="500" alt>
</p>

## COLMAP

[COLMAP](https://github.com/colmap/colmap) is a classical tool to perform SfM, meaning building a 3D model from a video or a bunch of images of a static object.

Create your directory, extract ~200-300 images from your video to a `images/` folder. Then run :

```
colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model OPENCV

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
```

After running the incremental mapper, you will have in the `sparse/0/` folder:

* `cameras.bin` corresponding to the intrinsic camera matrix (fx, fy, cx, cy, k1, k2, p1, p2)
* `images.bin` corresponding to the poses of each images (position + rotation)
* `points3D.bin` corresponding to the 3D points.

Using a points3D.bin is not very convenient, you can convert it to a PLY file with:

```
python3 convert_colmap_to_ply.py path/to/colmap/points3D.bin path/to/colmap/points.ply
```

Then you can visualize it with any pointcloud visualizer (CloudCompare, Blender, Meshlab etc). Then you can clean it a bit with SOR and crop it to focus only on the object you want.

## Keypoints selection and dataset generation

With the PLY, you have to select 20 Points of Interest (PoI). I recommend to take widely spread across your 3D points, ideally on all "faces" of your object. As you can guess, all PoI can not be seen on every images, sometimes there are hidden, sometimes there are out of bounds of the image. This is why I recommend to voxelize your space and check if your point is hidden or not.

Once you picked up the points, write them to a simple text file `points.txt` with xyz position on each line.

Then, thanks to the intrinsics and extrinsics contained in the colmap folder, you can project the keyponts:

```
python3 display_key_points.py path/to/colmap/ points.txt
```

<p>
<em>Points of interest / blue = visible / red = hidden)</em></br>
<img src="res/keypoints.gif"  width="500" alt>
</p>

## Neural network training

This is largely inspired from Centernet idea. Resnet-18 is used as a backbone, then a few convolutionnal layers are added to the top of it, to predict a heatmap and and offset map. In the end, the predictor works quite well. The big advantage to use a CNN is that you can have something very robust to lighting conditions or weather changes. You can easily do some data augmentation. Of course, we only predict visible points positions !

See [detect_multi_keypoints_20.ipynb](detect_multi_keypoints_20.ipynb) to access traning/inference. At the end, you can export a `keypoints.json`

In the end, the OpenCV Perspective-n-Point algorithm is used to recover pose from 2D PoI positions and their corresponding 3D world positions. Ransac is used to reject outliers. 

```
python3 pnp.py path/to/colmap/ video.mp4 inference_keypoints.json
```

When not enough points are seen or if the PnP didnt work, you will classify the frame as "Lost".

## What is the point ?

Well, once you have the pose of the camera with respect to your 3D point cloud, you can project the COLMAP point cloud to check if it works correctly. Then, you can display content for an AR application !

