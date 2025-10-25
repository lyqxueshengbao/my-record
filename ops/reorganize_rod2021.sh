# download all zip files and unzip
unzip -o TRAIN_RAD_H.zip
unzip -o TRAIN_CAM_0.zip
unzip -o TEST_RAD_H.zip
unzip -o TRAIN_RAD_H_ANNO.zip
unzip -o CAM_CALIB.zip


# make folders for data and annotations
mkdir sequences
mkdir annotations

# rename unzipped folders
mv TRAIN_RAD_H sequences/train
mv TRAIN_CAM_0 train
mv TEST_RAD_H sequences/test
mv TRAIN_RAD_H_ANNO annotations/train

# merge folders and remove redundant
rsync -av train/ sequences/train/
rm -r train