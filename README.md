# 5220-Final-Project

To download the imagenette dataset and crop the images to the appropriate size for VGG16:
```console
cd $SCRATCH
mkdir imagenette
cd imagenette
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xvzf imagenette2.tgz
module load pytorch/2.0.1
python crop.py
```
