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

To compile vgg.cu, 
```
module load cudnn
nvcc vgg.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -lcublas
```

build> cmake -D CMAKE_INSTALL_PREFIX=$HOME/opencv_install ..
make install -j

export LD_LIBRARY_PATH=/path/to/opencv/lib:$LD_LIBRARY_PATH

COMPILE THIS:

g++ vgg.cpp -I ../opencv_install/include/opencv4/ -L ../opencv_install/lib64/ -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
