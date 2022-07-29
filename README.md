# CertainNet_KITTI

<u><b>In case you are using docker</b></u>

docker volume create VOLUME_NAME

docker run -itd --gpus all --mount source=VOLUME_NAME,target=/home/USER DOCKER_IMAGE_ID

docker exec -it DOCKER_CONTAINER_ID /bin/bash

apt-get update

apt-get install wget git ffmpeg libsm6 libxext6 unzip nano -y

<u><b>Install miniconda and create new env</b></u>

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sh Miniconda3-latest-Linux-x86_64.sh

conda create --name ENV_NAME python=3.8

conda activate ENV_NAME

<u><b>Install torch based on cuda RUNTIME version in the machine (use nvcc --version to check). Requires CUDA 10.2 or 11.x (x<=4) and torch 1.11</b></u>

<u>CUDA 10.2</u>

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

<u>CUDA 11.3</u>

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

<u><b>Install DCNv2</b></u>

git clone https://github.com/jinfagang/DCNv2_latest.git

cd DCNv2_latest

python3 setup.py build develop

python3 testcuda.py

<u><b>Install dependencies</b></u>

python -m pip install opencv-python matplotlib numba pycocotools tqdm progress

<u><b>Clone this repo</b></u>

git clone https://github.com/qwertyman30/CenterNet_KITTI.git

<u><b>Setup the dataset</b></u>

cd CenterNet_KITTI/data/kitti

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

unzip data_object_image_2

unzip data_object_label_2

unzip data_object_calib

mkdir images && cd images

ln -s ../training/image_2 trainval

<u><b>Check the configs in opt dictionary. Make any changes. Run</b></u>

python CenterNet-Train-duq.py
