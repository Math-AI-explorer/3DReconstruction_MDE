REC_PATH=scene_reorganised/reconstruction

rm -rf $REC_PATH/dense/sparse2
mkdir $REC_PATH/dense/sparse2

colmap model_converter \
    --input_path $REC_PATH/dense/sparse \
    --output_path $REC_PATH/dense/sparse2 \
    --output_type TXT