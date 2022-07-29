# CertainNet_KITTI

In case you are using docker
<code><br>
docker volume create VOLUME_NAME
<br>
docker run -itd --gpus all --mount source=VOLUME_NAME,target=/home/USER DOCKER_IMAGE_ID
<br>
docker exec -it DOCKER_CONTAINER_ID /bin/bash
<br>
apt-get update
<br>
apt-get install wget git ffmpeg libsm6 libxext6 unzip nano -y
<br>
</code>

Install miniconda and create new env
<code><br>wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
<br>
sh Miniconda3-latest-Linux-x86_64.sh
<br>
conda create --name ENV_NAME python=3.8
<br>
conda activate ENV_NAME
</code>

install torch based on cuda RUNTIME version in the machine (use nvcc --version to check)

<u>CUDA 10.2</u>
<code><br>
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
</code>
<br><u>CUDA 11.3</u>
<code>
<br>conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
</code>

Install DCNv2
<code><br>
git clone https://github.com/jinfagang/DCNv2_latest.git
<br>
cd DCNv2_latest
<br>
python3 setup.py build develop
<br>
python3 testcuda.py
</code>

Install dependencies
<code><br>
python -m pip install opencv-python matplotlib numba pycocotools tqdm progress
</code>

Clone this repo
<code><br>
git clone https://github.com/qwertyman30/CenterNet_KITTI.git
</code>

Setup the dataset
<code><br>
cd CenterNet_KITTI/data/kitti
<br>
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
<br>
unzip data_object_image_2
<br>
unzip data_object_label_2
<br>
unzip data_object_calib
<br>
mkdir images && cd images
<br>
ln -s ../training/image_2 trainval
</code>

Check the configs in opt dictionary. Make any changes. Run
<code><br>python CenterNet-Train-duq.py</code>
