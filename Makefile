all:
	nvcc vgg.cu -O3 -g -std=c++20 -arch=sm_70 -I{$CUDNN_DIR}/include -L{$CUDNN_DIR}/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64

test:
	nvcc vgg16cnn.cu -O3 -g -std=c++20 -arch=sm_70 -I$CUDNN_DIR/include -L$CUDNN_DIR/lib -lcudnn -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/lib64
