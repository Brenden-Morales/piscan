mkdir -p sparse_bin/0 workspace export
# initialize a COLMAP database
touch database.db

colmap model_converter \
--input_path . \
--output_path sparse_bin/0 \
--output_type BIN

colmap feature_extractor \
--database_path database.db \
--image_path captures

colmap exhaustive_matcher \
--database_path database.db

colmap image_registrator \
--database_path database.db \
--input_path sparse_bin/0 \
--output_path sparse_bin/0

colmap point_triangulator \
--database_path database.db \
--input_path sparse_bin/0 \
--output_path sparse_bin/0 \
--image_path captures

colmap image_undistorter \
--image_path    captures \
--input_path    sparse_bin/0 \
--output_path   workspace \
--output_type   COLMAP \
--max_image_size 2000

InterfaceCOLMAP \
--input-file   workspace \
--image-folder images \
--output-file  scene.mvs

DensifyPointCloud \
--input-file  scene.mvs \
--output-file scene_dense.mvs \
--number-views 2 \
--number-views-fuse 2

meshlabserver \
-i scene_dense.ply \
-o scene_dense_ascii.ply \
-m vc vn