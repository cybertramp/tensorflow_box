# download and convert flowers data

export DATA_DIR=./flowers/dataset
export TRAIN_DIR=./flowers/train

python3 /projects/models/research/slim/download_and_convert_data.py --dataset_name=flowers --dataset_dir=${DATA_DIR}
