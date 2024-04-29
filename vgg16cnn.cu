#include <chrono>
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <numeric>
#include <vector>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


namespace fs = std::filesystem;

const int DIM = 224;
const int NUM_LAYERS = 3;
const int BATCH_SIZE = 32;
cudnnHandle_t cudnn_handle;
cublasHandle_t handle;


int find_arg_idx(int argc, char** argv, fs::path& p, fs::path& o) {
    if (argc < 2 || argc%2 == 1 || argc > 4 || strcmp(argv[1], "-h") == 0) {
      return -1;
    }
    int returnVal = 1;
    p = fs::path(argv[1]);
    for (int i = 2; i < argc; ++i) {
      if (strcmp(argv[i], "-o") == 0) {
        if (i != 2) return -1;
        else returnVal++;
      } else { // should be a path
        o = fs::path(argv[i]);
        returnVal++;
      }
    }
    return returnVal;
}

// From ZFTurbo
void read_image(char *in_file, float*** image) {
	int i, j, l;
	FILE *iin;
	float dval;
	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}
	/* Reading image */
	for (i = 0; i < DIM; i++) {
		for (j = 0; j < DIM; j++) {
			for (l = 0; l < 3; l++) {
				fscanf(iin, "%f", &dval);
				image[l][i][j] = dval;
			}
		}
	}
	fclose(iin);
}

struct Tensor4d
{
    cudnnTensorDescriptor_t desc;
    void *data;
    size_t data_size;

    Tensor4d(int n, int c, int h, int w)
    {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   n, c, h, w);
        data_size = n * c * h * w;
        cudaMalloc((void **)&data, data_size * sizeof(float));
    }
    ~Tensor4d()
    {
        cudaFree(data);
    }
};

struct Bias4d
{
    cudnnTensorDescriptor_t desc;
    void *data;
    size_t data_size;

    Bias4d(int n, int c, int h, int w)
    {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   n, c, h, w);
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }
        cudaMalloc((void **)&data, data_size * sizeof(float));
        auto code = cudaMemcpy(data, tmp, data_size * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    ~Bias4d()
    {
        cudaFree(data);
    }
};

struct Filter4d
{
    cudnnFilterDescriptor_t desc;
    void *data;
    size_t data_size;

    Filter4d(int n, int c, int h, int w)
    {
        cudnnCreateFilterDescriptor(&desc);
        cudnnSetFilter4dDescriptor(desc,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_TENSOR_NCHW,
                                   n, c, h, w);
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }

        cudaMalloc((void **)&data, data_size * sizeof(float));
        auto code = cudaMemcpy(data, tmp, data_size * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    ~Filter4d()
    {
        cudaFree(data);
    }
};

struct zeros
{
    void *data;
    size_t data_size;
    zeros(std::vector<int> dims) {
        data_size = std::accumulate(dims.begin(),
                                    dims.end(),
                                    1,
                                    std::multiplies<int>());
        std::vector<float> host_data(data_size);
        for (int i = 0; i < data_size; i++)
            host_data[i] = 0;

        cudaMalloc((void **)&data, data_size * sizeof(float));
        cudaMemcpy(data, host_data.data(), data_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    };
    ~zeros()
    {
        cudaFree(data);
    };
};

void cuConv2D(float *input, float *output, int w, int h, int c, int n, int k,
              int filter_w, int filter_h, int dilation_w, int dilation_h,
              int pad_w, int pad_h, int wstride, int hstride)
{

    size_t fwd_workspace_size;
    cudnnConvolutionFwdAlgo_t fwd_algo;

    const float alpha = 1.f;
    const float beta = 0.f;

    // datatype
    cudnnDataType_t dataType;
    dataType = CUDNN_DATA_FLOAT;
    
    // convolution mode
    cudnnConvolutionMode_t mode;
    mode = CUDNN_CONVOLUTION;

    int out_h, out_w, out_c, out_n;
    std::vector<int> output_dims_;

    // create conv desc
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    pad_h,
                                    pad_w,
                                    hstride,
                                    wstride,
                                    dilation_w,
                                    dilation_h,
                                    mode,
                                    dataType);

    // tensor desc
    Tensor4d x_desc(n, c, h, w);

    auto code = cudaMemcpy(x_desc.data, input, x_desc.data_size * sizeof(float),
                           cudaMemcpyHostToDevice);

    // filter desc
    Filter4d w_desc(k, c, filter_w, filter_h);
    
    // get conv dim
    cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                          x_desc.desc,
                                          w_desc.desc,
                                          &out_n,
                                          &out_c,
                                          &out_h,
                                          &out_w);

    Tensor4d h_desc(out_n, out_c, out_h, out_w);

    // choose forward algorith
    const int requestAlgoCount = 1;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;

    cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
                                         x_desc.desc,
                                         w_desc.desc,
                                         conv_desc,
                                         h_desc.desc,
                                         requestAlgoCount,
                                         &returnedAlgoCount,
                                         &perfResults);
    
    // what algorithm is choosed 
    fwd_algo = perfResults.algo;

    // get workspace size
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                            x_desc.desc,
                                            w_desc.desc,
                                            conv_desc,
                                            h_desc.desc,
                                            fwd_algo,
                                            &fwd_workspace_size);

    std::vector<int> u = std::vector<int>{static_cast<int>(fwd_workspace_size / sizeof(float)), 1};

    // init workspace
    zeros fwd_workspace(u);

    cudnnActivationDescriptor_t activationDesc;
    cudnnCreateActivationDescriptor(&activationDesc);
    auto code3 = cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 100);


    Bias4d bias(k, c, w, h);

    auto start = std::chrono::steady_clock::now();
    
    // fwd conv
    auto code2 = cudnnConvolutionBiasActivationForward(cudnn_handle,
                                                       &alpha,
                                                       x_desc.desc,
                                                       x_desc.data,
                                                       w_desc.desc,
                                                       w_desc.data,
                                                       conv_desc,
                                                       fwd_algo,
                                                       fwd_workspace.data,
                                                       fwd_workspace_size,
                                                       &beta,
                                                       h_desc.desc,
                                                       h_desc.data,
                                                       bias.desc,
                                                       bias.data,
                                                       activationDesc,
                                                       h_desc.desc,
                                                       h_desc.data);

    code = cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                                          std::micro>(end - start)
                                        .count());

    std::cout << " " << fwd_time << " ms" << std::endl;

    code = cudaMemcpy(output, h_desc.data, h_desc.data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // destroy conv desc
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return;
}

void cuMaxPool(float *input, float *output, int w, int h, int c, int n)
{
    const float alpha = 1.f;
    const float beta = 0.f;

    // create pool desc
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnCreatePoolingDescriptor(&pooling_desc);
    cudnnSetPooling2dDescriptor(
        pooling_desc,            //descriptor handle
        CUDNN_POOLING_MAX,       //mode - max pooling
        CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
        2,                       //window height
        2,                       //window width
        0,                       //vertical padding
        0,                       //horizontal padding
        2,                       //vertical stride
        2);                      //horizontal stride

    // tensor desc
    Tensor4d x_desc(n, c, h, w);

    auto code = cudaMemcpy(x_desc.data, input, x_desc.data_size * sizeof(float),
                           cudaMemcpyHostToDevice);

    Tensor4d h_desc(n, c, h / 2, w / 2);

    // std::cout << "outdim: " << w / 2 << ", " << h / 2 << ", " << c << ", " << n << std::endl;

    auto start = std::chrono::steady_clock::now();
    // fwd pool
    cudnnPoolingForward(
        cudnn_handle, //cuDNN context handle
        pooling_desc, //pooling descriptor handle
        &alpha,       //alpha scaling factor
        x_desc.desc,  //input tensor descriptor
        x_desc.data,  //input data pointer to GPU memory
        &beta,        //beta scaling factor
        h_desc.desc,  //output tensor descriptor
        h_desc.data); //output data pointer from GPU memory
    code = cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                                          std::micro>(end - start)
                                        .count());

    std::cout << " " << fwd_time << " ms" << std::endl;

    code = cudaMemcpy(output, h_desc.data, h_desc.data_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void cuFC(float *input, float *output, int left, int right)
{
    int lda = 1, ldb = left, ldc = 1, m = 1, k = left, n = right;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    float *h_B = (float *)malloc(left * right * sizeof(float));
    for (int i = 0; i < left * right; i++)
    {
        h_B[i] = (float)std::rand() / RAND_MAX / 1000;
    }

    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, left * sizeof(float));
    cudaMalloc(&d_B, left * right * sizeof(float));
    cudaMalloc(&d_C, right * sizeof(float));

    cudaMemcpy(d_A, input, left * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, left * right * sizeof(float), cudaMemcpyHostToDevice);

    // Create a handle for CUBLAS

    auto start = std::chrono::steady_clock::now();

    // Do the actual multiplication
    auto code = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double,
                                    std::micro>(end - start).count());

    std::cout << " " << fwd_time << " ms" << std::endl;

    cudaMemcpy(output, d_C, right * sizeof(float), cudaMemcpyDeviceToHost);
    // Destroy the handle
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main(int argc, char** argv) {
    fs::path base_dir;
    fs::path output_dir;
    // Parse input
    if (find_arg_idx(argc, argv, base_dir, output_dir) < 0) {
      std::cout << "Usage: <program> <images/weights directory> -o[optional] <path to output directory>" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "-h: see this help" << std::endl;
      std::cout << "-o <path>: path to output directory" << std::endl;
      return 0;
    }

    // Train

    // =============== Forward Propagation =====================

    std::srand(std::time(0));

    checkCUDNN(cudnnCreate(&cudnn_handle));
    checkCUDNN(cublasCreate(&handle));

    float *input;
    float *output;

    int data_size = 224 * 224 * 3 * 1;
    input = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; i++)
    {
        input[i] = (float)std::rand() / RAND_MAX;
    }

    // ===============  1 =====================
    std::cout << "CONV 224x224x64";
    output = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D(input, output, 224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 224x224x64";
    output = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D(input, output, 224, 224, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 112x112x64";
    output = (float *)malloc(112 * 112 * 64 * sizeof(float));
    cuMaxPool(input, output, 224, 224, 64, 1);
    std::swap(input, output);
    free(output);

    // ===============  2 =====================
    std::cout << "CONV 112x112x128";
    output = (float *)malloc(112 * 112 * 128 * 1 * sizeof(float));
    cuConv2D(input, output, 112, 112, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 112x112x128";
    output = (float *)malloc(112 * 112 * 128 * 1 * sizeof(float));
    cuConv2D(input, output, 112, 112, 128, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 56x56x128";
    output = (float *)malloc(56 * 56 * 128 * sizeof(float));
    cuMaxPool(input, output, 112, 112, 128, 1);
    std::swap(input, output);
    free(output);

    // ===============  3 =====================
    std::cout << "CONV 56x56x256";
    output = (float *)malloc(56 * 56 * 256 * 1 * sizeof(float));
    cuConv2D(input, output, 56, 56, 128, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 56x56x256";
    output = (float *)malloc(56 * 56 * 256 * 1 * sizeof(float));
    cuConv2D(input, output, 56, 56, 256, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 28x28x256";
    output = (float *)malloc(28 * 28 * 256 * sizeof(float));
    cuMaxPool(input, output, 56, 56, 256, 1);
    std::swap(input, output);
    free(output);

    // ===============  4 =====================
    std::cout << "CONV 28x28x512";
    output = (float *)malloc(28 * 28 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 28, 28, 256, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 28x28x512";
    output = (float *)malloc(28 * 28 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 28, 28, 512, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 14x14x512";
    output = (float *)malloc(14 * 14 * 512 * sizeof(float));
    cuMaxPool(input, output, 28, 28, 512, 1);
    std::swap(input, output);
    free(output);

    // ===============  5 =====================
    std::cout << "CONV 14x14x1024";
    output = (float *)malloc(14 * 14 * 1024 * 1 * sizeof(float));
    cuConv2D(input, output, 14, 14, 512, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 14x14x1024";
    output = (float *)malloc(14 * 14 * 1024 * 1 * sizeof(float));
    cuConv2D(input, output, 14, 14, 1024, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 7x7x1024";
    output = (float *)malloc(7 * 7 * 1024 * sizeof(float));
    cuMaxPool(input, output, 14, 14, 1024, 1);
    std::swap(input, output);
    free(output);

    // ===============  6 =====================
    std::cout << "FC 4096";
    output = (float *)malloc(4096 * sizeof(float));
    cuFC(input, output, 7 * 7 * 1024, 4096);
    std::swap(input, output);
    free(output);

    std::cout << "FC 4096";
    output = (float *)malloc(4096 * sizeof(float));
    cuFC(input, output, 4096, 4096);
    std::swap(input, output);
    free(output);

    std::cout << "FC 1000";
    output = (float *)malloc(1000 * sizeof(float));
    cuFC(input, output, 4096, 1000);

    // Backward Propagation

    





    

    
}