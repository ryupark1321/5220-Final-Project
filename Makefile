all:
	nvcc vgg.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -lcublas

test:
	nvcc vgg16cnn.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64

withopencv:
	nvcc vgg.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -I../opencv_install/include/opencv4/ -L ../opencv_install/lib64/ -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -lcublas -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc