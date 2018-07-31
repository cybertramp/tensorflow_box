export DATA_DIR=/projects/slim/flowers/dataset
export TRAIN_DIR=/projects/slim/flowers/train

python3 /projects/models/research/slim/train_image_classifier.py --train_dir=${TRAIN_DIR} --dataset_dir=${DATA_DIR} --dataset_name=flowers --dataset_split_name=train

