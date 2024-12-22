# The project folder must contain a folder "images" with all the images.
DATASET_PATH=scene_reorganised/images_adjusted
REC_PATH=scene_reorganised/reconstruction
DATABASE_PATH=$REC_PATH/database.db
COLMAP_EXE_PATH=colmap

# надо постоянно указывать путь до версии cuda,
# иначе nvcc не будет работать, как следствие и colmap
# export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

rm -rf $REC_PATH
mkdir $REC_PATH

LIBGL_ALWAYS_SOFTWARE=1 colmap feature_extractor \
    --database_path $DATABASE_PATH \
    --image_path $DATASET_PATH

colmap exhaustive_matcher \
    --database_path $DATABASE_PATH

mkdir $REC_PATH/sparse

colmap mapper \
    --database_path $DATABASE_PATH \
    --image_path $DATASET_PATH \
    --output_path $REC_PATH/sparse

mkdir $REC_PATH/dense

colmap image_undistorter \
    --image_path $DATASET_PATH \
    --input_path $REC_PATH/sparse/0 \
    --output_path $REC_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $REC_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $REC_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $REC_PATH/dense/fused.ply

# colmap poisson_mesher \
#   --input_path $DATASET_PATH/dense/fused.ply \
#   --output_path $DATASET_PATH/dense/meshed-poisson.ply

# colmap delaunay_mesher \
#   --input_path $DATASET_PATH/dense \

# git clone https://github.com/colmap/colmap.git
# cd colmap
# mkdir build
# cd build
# cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=75
# sudo chown -R $(whoami) .
# ninja -j1 
# sudo ninja install