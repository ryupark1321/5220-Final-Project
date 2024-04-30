all:
	nvcc vgg.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -lcublas

test:
	nvcc vgg16cnn.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64

withopencv:
	nvcc vgg16cnn.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64 -lcublas -I$OPENCV_DIR/include/opencv4 -L$OPENCV_DIR/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
