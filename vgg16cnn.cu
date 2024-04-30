#include <algorithm>
#include <chrono>
#include <ctime>
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <string>
#include "error_util.h" // Contains error handling functions
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#define MATRIX_DATA_TYPE float
#define CUBLAS_GEMM cublasSgemm
#define CUBLAS_GEAM cublasSgeam
#define CUBLAS_GEMV cublasSgemv
#define CUBLAS_SCAL cublasSscal
#define LEARNING_RATE (0.01)

#define IMAGE_H (224)
#define IMAGE_W (224)
#define IMAGE_D (3)
#define IMAGE_SIZE (IMAGE_H * IMAGE_W * IMAGE_D)
#define BATCH_SIZE (1)

#define MSIZE(a) ((a) * sizeof(value_type))

typedef enum
{
  CONV_LAYER = 0,
  POOL_LAYER = 1,
  FC_LAYER = 2,
  ACT_LAYER = 3,
  NORM_LAYER = 4,
  SOFTMAX_LAYER = 5
} LayerType;

#define BATCH_SIZE (32)
#define IMAGE_H (224)
#define IMAGE_W (224)
#define IMAGE_D (3)
#define DIM (224)
#define N (IMAGE_D * IMAGE_H * IMAGE_W)
#define DEBUG (0)

#define minn(a, b) (a < b ? a : b)
#define maxx(a, b) (a > b ? a : b)
#define minnn(a, b, c) (minn(minn(a, b), c))
#define maxxx(a, b, c) (maxx(maxx(a, b), c))

#define print(a) (std::cout << std::fixed << a)
#define println(a) (print(a << std::endl \
                            << std::flush))

#define ND_TENSOR_DESCRIPTOR

__global__ void FillOnes(MATRIX_DATA_TYPE *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
}

template <typename value_type> 
void printHostVector(std::string str, int size, value_type* vec){
    println(str<<" ("<<size<<") ");
    for (int i = 0; i < minn(size,400); i++)
    {
        print(vec[i] << " ");
    }
    println(" "); 
}

template <typename value_type>
void printDeviceVector(std::string str, int size, value_type* vec_d, int n=1)
{
    for (int i = 0; i < n; ++i)
    {    
        value_type* vec;
        vec = new value_type[size];
        cudaDeviceSynchronize();
        cudaMemcpy(vec, vec_d+i*size, MSIZE(size), cudaMemcpyDeviceToHost);
        printHostVector(str, size, vec);
        delete [] vec;
    }
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, MSIZE(size), cudaMemcpyDeviceToHost);
    std::cout.precision(5);
    std::cout.setf( std::ios::fixed, std::ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        print(value_type(vec[i]) << " ");
    }
    println(" ");
    delete [] vec;
}

void setTensorDesc(cudnnTensorDescriptor_t &tensorDesc,
                   cudnnTensorFormat_t &tensorFormat,
                   cudnnDataType_t &dataType,
                   int n,
                   int c,
                   int h,
                   int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
  checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
                                        tensorFormat,
                                        dataType,
                                        n, c,
                                        h,
                                        w));
#elif defined(ND_TENSOR_DESCRIPTOR)
  const int nDims = 4;
  int dimA[nDims] = {n, c, h, w};
  int strideA[nDims] = {c * h * w, h * w, w, 1};
  checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc,
                                        dataType,
                                        4,
                                        dimA,
                                        strideA));
#else
  checkCUDNN(cudnnSetTensor4dDescriptorEx(tensorDesc,
                                          dataType,
                                          n, c,
                                          h, w,
                                          c * h * w, h * w, w, 1));
#endif
}

#define NETWORK_ARCH                                                                                                                   \
  Layer_t<value_type> conv1;                                                                                                           \
  conv1.initConvLayer("conv1", /* inputs */ 3, /* outputs */ 64, /* kernel dim */ 3, /* stride */ 1, IMAGE_H, IMAGE_W, 0, BATCH_SIZE); \
  Layer_t<value_type> conv1act;                                                                                                        \
  conv1act.initActLayer("conv1act", conv1.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> conv2;                                                                                                           \
  conv2.initConvLayer("conv2", /* inputs */ 64, /* outputs */ 64, /* kernel dim */ 3, /* stride */ 1, conv1.out_height, conv1.out_width, 0, BATCH_SIZE); \
  Layer_t<value_type> conv2act;                                                                                                        \
  conv2act.initActLayer("conv2act", conv2.outputs, BATCH_SIZE);                                                                       \
  Layer_t<value_type> pool1;                                                                                                           \
  pool1.initPoolLayer("pool1", 2, 2, conv2, BATCH_SIZE);                                                                               \
  Layer_t<value_type> conv3;                                                                                                           \
  conv3.initConvLayer("conv3", pool1.kernel_dim, 128, 3, 1, pool1.out_width, pool1.out_height, pool1.outputs, BATCH_SIZE);             \
  Layer_t<value_type> conv3act;                                                                                                        \
  conv3act.initActLayer("conv3act", conv3.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> conv4;                                                                                                           \
  conv4.initConvLayer("conv4", pool1.kernel_dim, 128, 3, 1, pool1.out_width, pool1.out_height, pool1.outputs, BATCH_SIZE);             \
  Layer_t<value_type> conv4act;                                                                                                        \
  conv4act.initActLayer("conv4act", conv4.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> pool2;                                                                                                           \
  pool2.initPoolLayer("pool2", 2, 2, conv4, BATCH_SIZE);                                                                               \
  Layer_t<value_type> conv5;                                                                                                           \
  conv2.initConvLayer("conv5", pool2.kernel_dim, 256, 3, 1, pool2.out_width, pool2.out_height, pool2.outputs, BATCH_SIZE);             \
  Layer_t<value_type> conv5act;                                                                                                        \
  conv3act.initActLayer("conv5act", conv5.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> conv6;                                                                                                           \
  conv2.initConvLayer("conv6", pool2.kernel_dim, 256, 3, 1, pool2.out_width, pool2.out_height, pool2.outputs, BATCH_SIZE);             \
  Layer_t<value_type> conv6act;                                                                                                        \
  conv3act.initActLayer("conv6act", conv6.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> conv7;                                                                                                           \
  conv2.initConvLayer("conv7", pool2.kernel_dim, 256, 3, 1, pool2.out_width, pool2.out_height, pool2.outputs, BATCH_SIZE);             \
  Layer_t<value_type> conv7act;                                                                                                        \
  conv3act.initActLayer("conv7act", conv7.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> pool3;                                                                                                           \
  pool3.initPoolLayer("pool3", 2, 2, conv7, BATCH_SIZE);                                                                               \
  Layer_t<value_type> conv8;                                                                                                           \
  conv2.initConvLayer("conv8", pool3.kernel_dim, 512, 3, 1, pool3.out_width, pool3.out_height, pool3.outputs, BATCH_SIZE);             \
  Layer_t<value_type> conv8act;                                                                                                        \
  conv8act.initActLayer("conv8act", conv8.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> conv9;                                                                                                           \
  conv2.initConvLayer("conv9", pool3.kernel_dim, 512, 3, 1, pool3.out_width, pool3.out_height, pool3.outputs, BATCH_SIZE);             \
  Layer_t<value_type> conv9act;                                                                                                        \
  conv9act.initActLayer("conv9act", conv9.outputs, BATCH_SIZE);                                                                        \
  Layer_t<value_type> conv10;                                                                                                          \
  conv2.initConvLayer("conv10", pool3.kernel_dim, 512, 3, 1, pool3.out_width, pool3.out_height, pool3.outputs, BATCH_SIZE);            \
  Layer_t<value_type> conv10act;                                                                                                       \
  conv10act.initActLayer("conv10act", conv10.outputs, BATCH_SIZE);                                                                     \
  Layer_t<value_type> pool4;                                                                                                           \
  pool4.initPoolLayer("pool4", 2, 2, conv10, BATCH_SIZE);                                                                              \
  Layer_t<value_type> conv11;                                                                                                          \
  conv2.initConvLayer("conv11", pool4.kernel_dim, 512, 3, 1, pool4.out_width, pool4.out_height, pool4.outputs, BATCH_SIZE);            \
  Layer_t<value_type> conv11act;                                                                                                       \
  conv11act.initActLayer("conv11act", conv11.outputs, BATCH_SIZE);                                                                     \
  Layer_t<value_type> conv12;                                                                                                          \
  conv2.initConvLayer("conv12", pool4.kernel_dim, 512, 3, 1, pool4.out_width, pool4.out_height, pool4.outputs, BATCH_SIZE);            \
  Layer_t<value_type> conv12act;                                                                                                       \
  conv12act.initActLayer("conv12act", conv12.outputs, BATCH_SIZE);                                                                     \
  Layer_t<value_type> conv13;                                                                                                          \
  conv2.initConvLayer("conv13", pool4.kernel_dim, 512, 3, 1, pool4.out_width, pool4.out_height, pool4.outputs, BATCH_SIZE);            \
  Layer_t<value_type> conv13act;                                                                                                       \
  conv13act.initActLayer("conv13act", conv13.outputs, BATCH_SIZE);                                                                     \
  Layer_t<value_type> pool5;                                                                                                           \
  pool5.initPoolLayer("pool5", 2, 2, conv13, BATCH_SIZE);                                                                              \
  Layer_t<value_type> fc1;                                                                                                             \
  fc1.initFCLayer("fc1", pool5.outputs, 4096, BATCH_SIZE);                                                                             \
  Layer_t<value_type> fc1act;                                                                                                          \
  fc1act.initActLayer("fc1act", fc1.outputs, BATCH_SIZE);                                                                              \
  Layer_t<value_type> fc2;                                                                                                             \
  fc2.initFCLayer("fc2", fc1act.outputs, 4096, BATCH_SIZE);                                                                            \
  Layer_t<value_type> fc2act;                                                                                                          \
  fc2act.initActLayer("fc2act", fc2.outputs, BATCH_SIZE);                                                                              \
  Layer_t<value_type> fc3;                                                                                                             \
  fc3.initFCLayer("fc3", fc2act.outputs, 1000, BATCH_SIZE);                                                                            \
  Layer_t<value_type> fc3act;                                                                                                          \
  fc3act.initSoftmaxLayer("fc3act", fc3.outputs, BATCH_SIZE);

#define LOAD_DATA (conv1.load() && conv2.load() && conv3.load() && conv4.load() && conv5.load() && conv6.load() && conv7.load() && conv8.load() && conv9.load() && conv10.load() && conv11.load() && conv12.load() && conv13.load() && fc1.load() && fc2.load() && fc3.load())

#define SAVE_DATA (conv1.save() && conv2.save() && conv3.save() && conv4.save() && conv5.save() && conv6.save() && conv7.save() && conv8.save() && conv9.save() && conv10.save() && conv11.save() && conv12.save() && conv13.save() && fc1.save() && fc2.save() && fc3.save())

#define COPY_DATA_TO_DEVICE  \
  conv1.copyDataToDevice();  \
  conv2.copyDataToDevice();  \
  conv3.copyDataToDevice();  \
  conv4.copyDataToDevice();  \
  conv5.copyDataToDevice();  \
  conv6.copyDataToDevice();  \
  conv7.copyDataToDevice();  \
  conv8.copyDataToDevice();  \
  conv9.copyDataToDevice();  \
  conv10.copyDataToDevice(); \
  conv11.copyDataToDevice(); \
  conv12.copyDataToDevice(); \
  conv13.copyDataToDevice(); \
  fc1.copyDataToDevice();    \
  fc2.copyDataToDevice();    \
  fc3.copyDataToDevice();

#define COPY_DATA_TO_HOST  \
  conv1.copyDataToHost();  \
  conv2.copyDataToHost();  \
  conv3.copyDataToHost();  \
  conv4.copyDataToHost();  \
  conv5.copyDataToHost();  \
  conv6.copyDataToHost();  \
  conv7.copyDataToHost();  \
  conv8.copyDataToHost();  \
  conv9.copyDataToHost();  \
  conv10.copyDataToHost(); \
  conv11.copyDataToHost(); \
  conv12.copyDataToHost(); \
  conv13.copyDataToHost(); \
  fc1.copyDataToHost();    \
  fc2.copyDataToHost();    \
  fc3.copyDataToHost();

#define LAYER_NAMES \
  conv1, conv1act, conv2, conv2act, pool1, conv3, conv3act, conv4, conv4act, pool2, conv5, conv5act, conv6, conv6act, conv7, conv7act, pool3, conv8, conv8act, conv9, conv9act, conv10, conv10act, pool4, conv11, conv11act, conv12, conv12act, conv13, conv13act, pool5, fc1, fc1act, fc2, fc2act, fc3, fc3act

#define LAYER_NAMES_WITH_TYPE         \
  Layer_t<value_type> &conv1,         \
      Layer_t<value_type> &conv1act,  \
      Layer_t<value_type> &conv2,     \
      Layer_t<value_type> &conv2act,  \
      Layer_t<value_type> &pool1,     \
      Layer_t<value_type> &conv3,     \
      Layer_t<value_type> &conv3act,  \
      Layer_t<value_type> &conv4,     \
      Layer_t<value_type> &conv4act,  \
      Layer_t<value_type> &pool2,     \
      Layer_t<value_type> &conv5,     \
      Layer_t<value_type> &conv5act,  \
      Layer_t<value_type> &conv6,     \
      Layer_t<value_type> &conv6act,  \
      Layer_t<value_type> &conv7,     \
      Layer_t<value_type> &conv7act,  \
      Layer_t<value_type> &pool3,     \
      Layer_t<value_type> &conv8,     \
      Layer_t<value_type> &conv8act,  \
      Layer_t<value_type> &conv9,     \
      Layer_t<value_type> &conv9act,  \
      Layer_t<value_type> &conv10,    \
      Layer_t<value_type> &conv10act, \
      Layer_t<value_type> &pool4,     \
      Layer_t<value_type> &conv11,    \
      Layer_t<value_type> &conv11act, \
      Layer_t<value_type> &conv12,    \
      Layer_t<value_type> &conv12act, \
      Layer_t<value_type> &conv13,    \
      Layer_t<value_type> &conv13act, \
      Layer_t<value_type> &pool5,     \
      Layer_t<value_type> &fc1,       \
      Layer_t<value_type> &fc1act,    \
      Layer_t<value_type> &fc2,       \
      Layer_t<value_type> &fc2act,    \
      Layer_t<value_type> &fc3,       \
      Layer_t<value_type> &fc3act

namespace fs = std::filesystem;

template <class value_type>
struct Layer_t
{
  LayerType layerType;
  std::string layername;
  int n; // batch_size

  int inputs, outputs, kernel_dim; // linear dimension (i.e. size is kernel_dim * kernel_dim)
  int w_size, b_size, d_size;

  int in_height, in_width;
  int out_height, out_width;

  value_type *data_h, *data_d;
  value_type *bias_h, *bias_d;

  value_type *output_d, *del_d;

  value_type *oneVec_d;

  // Convolutional Layer
  cudnnConvolutionDescriptor_t convDesc;
  cudnnTensorDescriptor_t convBiasTensorDesc;
  cudnnFilterDescriptor_t convFilterDesc;
  cudnnTensorDescriptor_t convSrcTensorDesc, convDstTensorDesc;
  cudnnConvolutionFwdAlgo_t convFwdAlgo;
  cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
  cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
  size_t convFwdSizeInBytes, convBwdDataSizeInBytes, convBwdFilterSizeInBytes;

  // Pooling Layer
  cudnnPoolingDescriptor_t poolDesc;
  cudnnTensorDescriptor_t poolSrcTensorDesc, poolDstTensorDesc;
  cudnnFilterDescriptor_t poolFilterDesc;
  int size, stride;

  // Fully Connected Layer

  // Activation Layer
  cudnnActivationDescriptor_t activDesc;
  cudnnTensorDescriptor_t actTensorDesc;

  // Normal Layer

  // Softmax Layer

  cudnnDataType_t dataType;
  cudnnTensorFormat_t tensorFormat;

  const std::string weights_folder = "bins/";
  double learning_rate;

  Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL),
              inputs(0), outputs(0), kernel_dim(0)
  {
    switch (sizeof(value_type))
    {
    case 4:
      dataType = CUDNN_DATA_FLOAT;
      break;
    case 8:
      dataType = CUDNN_DATA_DOUBLE;
      break;
    default:
      FatalError("Unsupported data type");
    }
    tensorFormat = CUDNN_TENSOR_NCHW;
    data_d = bias_d = output_d = del_d = NULL;
    oneVec_d = NULL;
    n = 0;
    convFwdSizeInBytes = convBwdDataSizeInBytes = convBwdFilterSizeInBytes = 0;
  };

  ~Layer_t()
  {
    if (data_h != NULL)
      delete[] data_h;
    if (bias_h != NULL)
      delete[] bias_h;

    if (data_d != NULL)
      checkCudaErrors(cudaFree(data_d));
    if (bias_d != NULL)
      checkCudaErrors(cudaFree(bias_d));
    if (output_d != NULL)
      checkCudaErrors(cudaFree(output_d));
    if (del_d != NULL)
      checkCudaErrors(cudaFree(del_d));

    if (layerType == CONV_LAYER)
    {
      destroyConvLayer();
    }
    else if (layerType == POOL_LAYER)
    {
      destroyPoolLayer();
    }
    else if (layerType == ACT_LAYER || layerType == SOFTMAX_LAYER || layerType == NORM_LAYER)
    {
      destroyActLayer();
    }
    else if (layerType == FC_LAYER)
    {
      destroyLayer();
    }
  }

  void setHandles(int _n)
  {
    if (_n == n)
      return;
    n = _n;

    if (oneVec_d != NULL)
      checkCudaErrors(cudaFree(oneVec_d));
    checkCudaErrors(cudaMalloc(&oneVec_d, MSIZE(n)));

    FillOnes<<<1, n>>>(oneVec_d, n);

    if (layerType == CONV_LAYER)
    {
      createConvHandles();
    }
    else if (layerType == POOL_LAYER)
    {
      createPoolHandles();
    }
    else if (layerType == ACT_LAYER || layerType == SOFTMAX_LAYER || layerType == NORM_LAYER)
    {
      createActHandles();
    }
    else
    { // FC_LAYER
      createFCHandles();
    }
  }

  void createPoolHandles()
  {
    int c, h, w;
    c = kernel_dim;
    h = in_height;
    w = in_width;
    setTensorDesc(poolSrcTensorDesc, tensorFormat, dataType, n, c, h, w);

    println("pool in >> n:" << n << "\tc:" << c << "\th:" << h << "\tw:" << w);
    const int tensorDims = 4;
    int tensorOuputDimA[tensorDims] = {n, c, h, w};
    checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolDesc,
                                                 poolSrcTensorDesc,
                                                 tensorDims,
                                                 tensorOuputDimA));
    n = tensorOuputDimA[0];
    c = tensorOuputDimA[1];
    h = tensorOuputDimA[2];
    w = tensorOuputDimA[3];

    println("pool out >> n:" << n << "\tc:" << c << "\th:" << h << "\tw:" << w);
    out_height = h;
    out_width = w;

    setTensorDesc(poolDstTensorDesc, tensorFormat, dataType, n, c, h, w);

    b_size = kernel_dim * out_width * out_height;
    outputs = b_size;
    inputs = kernel_dim * in_width * in_height;

    if (output_d != NULL)
      checkCudaErrors(cudaFree(output_d));
    if (del_d != NULL)
      checkCudaErrors(cudaFree(del_d));

    checkCudaErrors(cudaMalloc(&output_d, MSIZE(n * outputs)));
    checkCudaErrors(cudaMalloc(&del_d, MSIZE(n * inputs)));
  }

  void createFCHandles()
  {
    if (output_d != NULL)
      checkCudaErrors(cudaFree(output_d));
    if (del_d != NULL)
      checkCudaErrors(cudaFree(del_d));

    checkCudaErrors(cudaMalloc(&output_d, MSIZE(n * outputs)));
    checkCudaErrors(cudaMalloc(&del_d, MSIZE(n * inputs)));
  }

  void createActHandles()
  {
    int c, h, w;
    h = w = 1;
    c = inputs;
    setTensorDesc(actTensorDesc, tensorFormat, dataType, n, c, h, w);

    if (output_d != NULL)
      checkCudaErrors(cudaFree(output_d));
    if (del_d != NULL)
      checkCudaErrors(cudaFree(del_d));

    checkCudaErrors(cudaMalloc(&output_d, MSIZE(n * outputs)));
    checkCudaErrors(cudaMalloc(&del_d, MSIZE(n * inputs)));
  }

  void createConvHandles()
  {
    int c = inputs;
    int h = in_height;
    int w = in_width;

    println("conv in >> n:" << n << "\tc:" << c << "\th:" << h << "\tw:" << w);
    checkCUDNN(cudnnSetTensor4dDescriptor(convSrcTensorDesc,
                                          tensorFormat,
                                          dataType,
                                          n, c,
                                          h, w));

    checkCUDNN(cudnnSetTensor4dDescriptor(convBiasTensorDesc,
                                          tensorFormat,
                                          dataType,
                                          1, outputs,
                                          1, 1));

    checkCUDNN(cudnnSetFilter4dDescriptor(convFilterDesc,
                                          dataType,
                                          tensorFormat,
                                          outputs, inputs,
                                          kernel_dim, kernel_dim));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                               0, 0,           //	padding
                                               stride, stride, //	stride
                                               1, 1,           // 	upscaling
                                               CUDNN_CROSS_CORRELATION, cudnnDataType_t::CUDNN_DATA_FLOAT));
    // Find dimension of convolution output
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                     convSrcTensorDesc,
                                                     convFilterDesc,
                                                     &n, &c, &h, &w));

    out_width = w;
    out_height = h;

    println("conv out >> n:" << n << "\tc:" << c << "\th:" << h << "\tw:" << w);
    checkCUDNN(cudnnSetTensor4dDescriptor(convDstTensorDesc,
                                          tensorFormat,
                                          dataType,
                                          n, c,
                                          h, w));
    cudnnHandle_t cudnnHandle;
    checkCUDNN(cudnnCreate(&cudnnHandle));

    convFwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    // checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
    //                                                convSrcTensorDesc,
    //                                                convFilterDesc,
    //                                                convDesc,
    //                                                convDstTensorDesc,
    //                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //                                                0,
    //                                                &convFwdAlgo));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                       convSrcTensorDesc,
                                                       convFilterDesc,
                                                       convDesc,
                                                       convDstTensorDesc,
                                                       convFwdAlgo,
                                                       &convFwdSizeInBytes));

    convBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    // checkCUDNN( cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
    // 													convFilterDesc,
    // 													convDstTensorDesc,
    // 													convDesc,
    // 													convSrcTensorDesc,
    // 													CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
    // 													0,
    // 													&convBwdDataAlgo));

    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                            convFilterDesc,
                                                            convDstTensorDesc,
                                                            convDesc,
                                                            convSrcTensorDesc,
                                                            convBwdDataAlgo,
                                                            &convBwdDataSizeInBytes));

    convBwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    // checkCUDNN( cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
    // 														convSrcTensorDesc,
    // 														convDstTensorDesc,
    // 														convDesc,
    // 														convFilterDesc,
    // 														CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
    // 														0,
    // 														&convBwdFilterAlgo));
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                              convSrcTensorDesc,
                                                              convDstTensorDesc,
                                                              convDesc,
                                                              convFilterDesc,
                                                              convBwdFilterAlgo,
                                                              &convBwdFilterSizeInBytes));

    // println("handles: "<<(int)convFwdAlgo<<" "<<(int)convBwdDataAlgo<<" "<<(int)convBwdFilterAlgo);

    checkCUDNN(cudnnDestroy(cudnnHandle));

    if (data_d != NULL)
      checkCudaErrors(cudaFree(data_d));
    if (bias_d != NULL)
      checkCudaErrors(cudaFree(bias_d));
    if (output_d != NULL)
      checkCudaErrors(cudaFree(output_d));
    if (del_d != NULL)
      checkCudaErrors(cudaFree(del_d));

    checkCudaErrors(cudaMalloc(&data_d, MSIZE(w_size)));
    checkCudaErrors(cudaMalloc(&bias_d, MSIZE(b_size)));
    checkCudaErrors(cudaMalloc(&output_d, MSIZE(n * outputs * out_height * out_width)));
    checkCudaErrors(cudaMalloc(&del_d, MSIZE(n * d_size)));
  }

  void initConvLayer(std::string _layername, int _inputs, int _outputs, int _kernel_dim, int _stride, int _in_height, int _in_width, int _d_size = 0, int _batch_size = 1)
  {
    layerType = CONV_LAYER;
    layername = _layername;
    inputs = _inputs;
    outputs = _outputs;
    kernel_dim = _kernel_dim;
    stride = _stride;
    in_width = _in_width;
    in_height = _in_height;
    w_size = inputs * outputs * kernel_dim * kernel_dim;
    b_size = outputs;
    d_size = _d_size;

    data_h = new value_type[w_size];
    bias_h = new value_type[b_size];

    // Random Initialization
    // TODO : Fix this random initialization
    for (int i = 0; i < w_size; i++)
      data_h[i] = (((value_type)rand()) / (rand() + 1)) / 100000;
    for (int i = 0; i < b_size; i++)
      bias_h[i] = (((value_type)rand()) / (rand() + 1)) / 100000;

    checkCUDNN(cudnnCreateTensorDescriptor(&convSrcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&convDstTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&convFilterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&convBiasTensorDesc));

    setHandles(_batch_size);

    copyDataToDevice();
  }

  void initPoolLayer(std::string _layername, int _size, int _stride, Layer_t<value_type> &conv, int _batch_size = 1)
  {
    layerType = POOL_LAYER;
    layername = _layername;
    size = _size;
    stride = _stride;
    w_size = 0;
    kernel_dim = conv.outputs;
    in_height = conv.out_height;
    in_width = conv.out_width;

    checkCUDNN(cudnnCreateTensorDescriptor(&poolSrcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&poolDstTensorDesc));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_PROPAGATE_NAN,
                                           size, size,
                                           0, 0,
                                           stride, stride));

    setHandles(_batch_size);
  }

  void initFCLayer(std::string _layername, int _inputs, int _outputs, int _batch_size = 1)
  {
    layerType = FC_LAYER;
    layername = _layername;
    inputs = _inputs;
    outputs = _outputs;
    kernel_dim = 1;
    w_size = inputs * outputs * kernel_dim * kernel_dim;
    b_size = outputs;

    data_h = new value_type[w_size];
    bias_h = new value_type[b_size];

    // Random Initialization
    // TODO : Fix this random initialization
    for (int i = 0; i < w_size; i++)
      data_h[i] = (((value_type)rand()) / (rand() + 1)) / 100000;
    for (int i = 0; i < b_size; i++)
      bias_h[i] = (((value_type)rand()) / (rand() + 1)) / 100000;

    checkCudaErrors(cudaMalloc(&data_d, MSIZE(w_size)));
    checkCudaErrors(cudaMalloc(&bias_d, MSIZE(b_size)));

    setHandles(_batch_size);

    copyDataToDevice();
  }

  void initActLayer(std::string _layername, int _outputs, int _batch_size = 1)
  {
    initLayer(_layername, ACT_LAYER, _outputs, _batch_size);
  }

  void initSoftmaxLayer(std::string _layername, int _outputs, int _batch_size = 1)
  {
    initLayer(_layername, SOFTMAX_LAYER, _outputs, _batch_size);
  }

  void destroyConvLayer()
  {
    checkCUDNN(cudnnDestroyTensorDescriptor(convSrcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(convDstTensorDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(convFilterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(convBiasTensorDesc));
  }

  void destroyPoolLayer()
  {
    checkCUDNN(cudnnDestroyTensorDescriptor(poolSrcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(poolDstTensorDesc));
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
  }

  void destroyActLayer()
  {
    checkCUDNN(cudnnDestroyActivationDescriptor(activDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(actTensorDesc));
  }

  void destroyLayer()
  {
  }

  void copyDataToDevice()
  {
    if (data_h != NULL)
      checkCudaErrors(cudaMemcpy(data_d, data_h, MSIZE(w_size), cudaMemcpyHostToDevice));
    if (bias_h != NULL)
      checkCudaErrors(cudaMemcpy(bias_d, bias_h, MSIZE(b_size), cudaMemcpyHostToDevice));
  }

  void copyDataToHost()
  {
    if (data_h != NULL)
      checkCudaErrors(cudaMemcpy(data_h, data_d, MSIZE(w_size), cudaMemcpyDeviceToHost));
    if (bias_h != NULL)
      checkCudaErrors(cudaMemcpy(bias_h, bias_d, MSIZE(b_size), cudaMemcpyDeviceToHost));
  }

  bool load()
  {
    std::string dtype = (sizeof(value_type) == 4 ? "_float_" : "_double_");
    return loadWeights(layername + dtype + "weights.bin", w_size, data_h) && loadWeights(layername + dtype + "bias.bin", b_size, bias_h);
  }

  bool save()
  {
    std::string dtype = (sizeof(value_type) == 4 ? "_float_" : "_double_");
    return saveWeights(layername + dtype + "weights.bin", w_size, data_h) && saveWeights(layername + dtype + "bias.bin", b_size, bias_h);
  }

  bool loadWeights(std::string filename, size_t size, value_type *matrix)
  {
    filename = weights_folder + filename;
    std::ifstream myfile(filename.c_str(), std::ios::in | std::ios::binary);
    if (myfile.is_open())
    {
      myfile.read((char *)matrix, MSIZE(size));
      return true;
    }
    else
    {
      println("Error reading file " << filename);
      return false;
    }
  }

  bool saveWeights(std::string filename, size_t size, value_type *matrix)
  {
    filename = weights_folder + filename;
    std::ofstream myfile(filename.c_str(), std::ios::out | std::ios::binary);
    if (myfile.is_open())
    {
      myfile.write((char *)matrix, MSIZE(size));
      return true;
    }
    else
    {
      println("Error saving file " << filename);
      return false;
    }
  }

private:
  void initLayer(std::string _layername, LayerType _layerType, int _outputs, int _batch_size = 1)
  {
    layerType = _layerType;
    layername = _layername;
    inputs = _outputs;
    outputs = _outputs;
    kernel_dim = 1;
    w_size = 0;
    b_size = 0;

    checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&actTensorDesc));
    checkCUDNN(cudnnSetActivationDescriptor(activDesc,
                                            CUDNN_ACTIVATION_RELU, // CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN,
                                            0.0));

    setHandles(_batch_size);
  }

  void readAllocInit(const char *fname, int size, value_type **data_h, value_type **data_d)
  {
    readAllocMemcpy<value_type>(fname, size, data_h, data_d);
  }
};

int find_arg_idx(int argc, char **argv, fs::path &p, fs::path &o)
{
  if (argc < 2 || argc % 2 == 1 || argc > 4 || strcmp(argv[1], "-h") == 0)
  {
    return -1;
  }
  int returnVal = 1;
  p = fs::path(argv[1]);
  for (int i = 2; i < argc; ++i)
  {
    if (strcmp(argv[i], "-o") == 0)
    {
      if (i != 2)
        return -1;
      else
        returnVal++;
    }
    else
    { // should be a path
      o = fs::path(argv[i]);
      returnVal++;
    }
  }
  return returnVal;
}

__global__ void getDiffDataD(MATRIX_DATA_TYPE *targets, MATRIX_DATA_TYPE *diffData, int label_count, int _batch_size)
{
  int idx = threadIdx.x;
  if (idx >= _batch_size)
    return;
  const int label_value = static_cast<int>(targets[idx]);
  diffData[idx * label_count + label_value] -= 1;
}

template <class value_type>
class network_t
{
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;
  value_type vOne, vZero;

  void createHandles()
  {
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCublasErrors(cublasCreate(&cublasHandle));
  }

  void destroyHandles()
  {
    checkCUDNN(cudnnDestroy(cudnnHandle));
    checkCublasErrors(cublasDestroy(cublasHandle));
  }

public:
  network_t()
  {
    vOne = value_type(1);
    vZero = value_type(0);
    createHandles();
  };

  ~network_t()
  {
    destroyHandles();
  }

  void resize(int size, value_type **data)
  {
    if (*data != NULL)
    {
      checkCudaErrors(cudaFree(*data));
    }
    checkCudaErrors(cudaMalloc(data, MSIZE(size)));
  }

  void addBias(const cudnnTensorDescriptor_t &convDstTensorDesc, Layer_t<value_type> &layer, int c, value_type *data)
  {
    checkCUDNN(cudnnAddTensor(cudnnHandle,
                              &vOne,
                              layer.convBiasTensorDesc,
                              layer.bias_d,
                              &vOne,
                              convDstTensorDesc,
                              data));
  }

  void fullyConnectedForward(Layer_t<value_type> &layer,
                             int &n,
                             value_type *srcData)
  {
    layer.setHandles(n);

    // int dim_x = layer.inputs;
    // int dim_y = layer.outputs;

    // checkCudaErrors( cudaMemcpy(layer.output_d, layer.bias_d, MSIZE(dim_y), cudaMemcpyDeviceToDevice) );

    // checkCublasErrors( CUBLAS_GEMV(cublasHandle, CUBLAS_OP_T,
    //                          dim_x, dim_y,
    //                          &vOne,
    //                          layer.data_d, dim_x,
    //                          srcData, 1,
    //                          &vOne,
    //                          layer.output_d, 1) );

    // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
    checkCudaErrors(CUBLAS_GEMM(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                layer.outputs, n, layer.inputs,
                                &vOne,
                                layer.data_d, layer.inputs,
                                srcData, layer.inputs,
                                &vZero,
                                layer.output_d, layer.outputs));
    // printDeviceVector("One Vector:\n", n, layer.oneVec_d);
    // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
    checkCudaErrors(CUBLAS_GEMM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                layer.outputs, n, 1,
                                &vOne,
                                layer.bias_d, layer.outputs,
                                layer.oneVec_d, 1,
                                &vOne,
                                layer.output_d, layer.outputs));
  }

  void convoluteForward(Layer_t<value_type> &layer,
                        int &n,
                        value_type *srcData)
  {
    layer.setHandles(n);

    if (DEBUG)
      printDeviceVector("Conv Weights:\n", layer.w_size, layer.data_d);
    if (DEBUG)
      printDeviceVector("Conv Bias:\n", layer.b_size, layer.bias_d);
    void *workSpace = NULL;
    if (layer.convFwdSizeInBytes != 0)
    {
      checkCudaErrors(cudaMalloc(&workSpace, layer.convFwdSizeInBytes));
    }
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                                       &vOne,
                                       layer.convSrcTensorDesc,
                                       srcData,
                                       layer.convFilterDesc,
                                       layer.data_d,
                                       layer.convDesc,
                                       layer.convFwdAlgo,
                                       workSpace,
                                       layer.convFwdSizeInBytes,
                                       &vZero,
                                       layer.convDstTensorDesc,
                                       layer.output_d));
    addBias(layer.convDstTensorDesc, layer, layer.outputs, layer.output_d);
    if (DEBUG)
      printDeviceVector("Conv Output:\n", layer.outputs * layer.out_height * layer.out_width, layer.output_d);
    if (layer.convFwdSizeInBytes != 0)
    {
      checkCudaErrors(cudaFree(workSpace));
    }
  }

  void convoluteBackward(Layer_t<value_type> &layer,
                         int &n,
                         value_type *diffData)
  {
    void *workSpace = NULL;
    if (layer.convBwdDataSizeInBytes != 0)
    {
      checkCudaErrors(cudaMalloc(&workSpace, layer.convBwdDataSizeInBytes));
    }
    checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
                                            &vOne,
                                            layer.convFilterDesc, layer.data_d,
                                            layer.convDstTensorDesc, diffData,
                                            layer.convDesc, layer.convBwdDataAlgo,
                                            workSpace, layer.convBwdDataSizeInBytes,
                                            &vZero,
                                            layer.convSrcTensorDesc, layer.del_d));
    if (layer.convBwdDataSizeInBytes != 0)
    {
      checkCudaErrors(cudaFree(workSpace));
    }
  }

  void poolForward(Layer_t<value_type> &layer,
                   int &n,
                   value_type *srcData)
  {
    layer.setHandles(n);

    if (DEBUG)
      printDeviceVector("Pooling Input:\n", layer.inputs, layer.output_d);
    checkCUDNN(cudnnPoolingForward(cudnnHandle,
                                   layer.poolDesc,
                                   &vOne,
                                   layer.poolSrcTensorDesc,
                                   srcData,
                                   &vZero,
                                   layer.poolDstTensorDesc,
                                   layer.output_d));
    if (DEBUG)
      printDeviceVector("Pooling Output:\n", layer.outputs, layer.output_d);
  }

  void poolBackward(Layer_t<value_type> &layer,
                    int &n,
                    value_type *diffData, value_type *srcData)
  {

    if (DEBUG)
      printDeviceVector("Pooling back Input: ", layer.outputs, srcData);
    checkCUDNN(cudnnPoolingBackward(cudnnHandle,
                                    layer.poolDesc,
                                    &vOne,
                                    layer.poolDstTensorDesc, layer.output_d,
                                    layer.poolDstTensorDesc, diffData,
                                    layer.poolSrcTensorDesc, srcData,
                                    &vZero,
                                    layer.poolSrcTensorDesc, layer.del_d));
    if (DEBUG)
      printDeviceVector("Pooling back Output: ", layer.inputs, layer.del_d);
  }

  void softmaxForward(Layer_t<value_type> &layer,
                      int &n, value_type *srcData)
  {
    layer.setHandles(n);
    checkCUDNN(cudnnSoftmaxForward(cudnnHandle,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &vOne,
                                   layer.actTensorDesc,
                                   srcData,
                                   &vZero,
                                   layer.actTensorDesc,
                                   layer.output_d));
  }

  void getDiffData(Layer_t<value_type> &layer, int target, value_type **diffData)
  {
    resize(layer.outputs, diffData);
    value_type outputh[layer.outputs];
    checkCudaErrors(cudaMemcpy(outputh, layer.output_d, MSIZE(layer.outputs), cudaMemcpyDeviceToHost));
    for (int i = 0; i < layer.outputs; i++)
    {
      if (i == target)
        outputh[i] -= 1;
    }
    checkCudaErrors(cudaMemcpy(*diffData, outputh, MSIZE(layer.outputs), cudaMemcpyHostToDevice));
  }

  void softmaxBackward(Layer_t<value_type> &layer,
                       int &n,
                       value_type *diffData, value_type *srcData)
  {
    checkCUDNN(cudnnSoftmaxBackward(cudnnHandle,
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &vOne,
                                    layer.actTensorDesc,
                                    layer.output_d,
                                    layer.actTensorDesc,
                                    diffData,
                                    &vZero,
                                    layer.actTensorDesc,
                                    layer.del_d));
  }

  void activationForward(Layer_t<value_type> &layer,
                         int &n, value_type *srcData)
  {
    layer.setHandles(n);
    checkCUDNN(cudnnActivationForward(cudnnHandle,
                                      layer.activDesc,
                                      &vOne,
                                      layer.actTensorDesc,
                                      srcData,
                                      &vZero,
                                      layer.actTensorDesc,
                                      layer.output_d));
  }

  void fullyConnectedBackward(Layer_t<value_type> &layer,
                              int &n, value_type *srcData)
  {
    // checkCudaErrors( CUBLAS_GEMV(cublasHandle, CUBLAS_OP_N,
    // 							  layer.inputs, layer.outputs,
    // 							  &vOne,
    // 							  layer.data_d, layer.inputs,
    // 							  srcData, 1,
    // 							  &vZero,
    // 							  layer.del_d, 1) );

    checkCudaErrors(CUBLAS_GEMM(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                layer.inputs, n, layer.outputs,
                                &vOne,
                                layer.data_d, layer.inputs,
                                srcData, layer.outputs,
                                &vZero,
                                layer.del_d, layer.inputs));
  }

  void activationBackward(Layer_t<value_type> &layer,
                          int &n,
                          value_type *srcDiffData, value_type *srcData)
  {
    checkCUDNN(cudnnActivationBackward(cudnnHandle,
                                       layer.activDesc,
                                       &vOne,
                                       layer.actTensorDesc,
                                       layer.output_d,
                                       layer.actTensorDesc,
                                       srcDiffData,
                                       layer.actTensorDesc,
                                       srcData,
                                       &vZero,
                                       layer.actTensorDesc,
                                       layer.del_d));
  }

  void fullyConnectedUpdateWeights(Layer_t<value_type> &layer, value_type *diffData, value_type *srcData, int n)
  {
    value_type *dstData = NULL;
    resize(layer.inputs * layer.outputs, &dstData);
    double learning_rate = LEARNING_RATE;
    value_type lr = value_type(-learning_rate); // learning rate

    // if (DEBUG) printDeviceVector("last_input: \n", layer.inputs, last_input);
    // if (DEBUG) printDeviceVector("del_W: \n", layer.outputs, layer.del_d);

    checkCudaErrors(CUBLAS_GEMM(cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                layer.inputs, layer.outputs, n,
                                &vOne,
                                srcData, layer.inputs,
                                diffData, layer.outputs,
                                &vZero,
                                dstData, layer.inputs));

    // if (DEBUG) printDeviceVector("\tdelta_W (del_W*hidden_input): \n", layer.inputs*layer.outputs, dstData);

    const value_type *B = layer.data_d;
    // C = α op ( A ) + β * C
    // C = 0.1 * delta_W2 + C
    // if (DEBUG) printDeviceVector("\tW = W + 0.1*delta_W: old\n", layer.inputs*layer.outputs, layer.data_d);

    checkCudaErrors(CUBLAS_GEAM(cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                layer.inputs, layer.outputs,
                                &lr,
                                dstData, layer.inputs,
                                &vOne,
                                B, layer.inputs,
                                layer.data_d, layer.inputs));
    // if (DEBUG) printDeviceVector("\tW: \n", layer.inputs*layer.outputs, layer.data_d);

    // printDeviceVector("\n yo \n", layer.outputs, diffData, n);
    // printDeviceVector("\n ones \n", n, layer.oneVec_d);
    resize(layer.outputs, &dstData);

    checkCudaErrors(CUBLAS_GEMV(cublasHandle,
                                CUBLAS_OP_N,
                                layer.outputs, n,
                                &vOne,
                                diffData, layer.outputs,
                                layer.oneVec_d, 1,
                                &vZero,
                                dstData, 1));
    // printDeviceVector("\n sum \n", layer.outputs, dstData);

    // place bias into dstData
    const value_type *B2 = layer.bias_d;
    // if (DEBUG) printDeviceVector("\tdel_W:\n", layer.outputs, layer.del_d);
    // if (DEBUG) printDeviceVector("\tB = B + 0.1*del_W: old\n", layer.outputs, layer.bias_d);
    checkCudaErrors(CUBLAS_GEAM(cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                1, layer.outputs,
                                &lr,
                                dstData, 1,
                                &vOne,
                                B2, 1,
                                layer.bias_d, 1));
    // if (DEBUG) printDeviceVector("\tB:\n", layer.outputs, layer.bias_d);

    checkCudaErrors(cudaFree(dstData));
  }

  void convolutionalUpdateWeights(Layer_t<value_type> &layer, value_type *diffData, value_type *srcData)
  {

    if (DEBUG)
      println("Convolutional Update Weights:");

    value_type *gconvB = NULL, *gconvW = NULL;
    resize(layer.outputs, &gconvB);
    resize(layer.w_size, &gconvW);

    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle,
                                            &vOne,
                                            layer.convDstTensorDesc, diffData,
                                            &vZero,
                                            layer.convBiasTensorDesc, gconvB));

    if (DEBUG)
      printDeviceVector(" gconvB: ", layer.outputs, gconvB);

    void *workSpace = NULL;

    if (layer.convBwdFilterSizeInBytes != 0)
    {
      checkCudaErrors(cudaMalloc(&workSpace, layer.convBwdFilterSizeInBytes));
    }
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,
                                              &vOne,
                                              layer.convSrcTensorDesc, srcData,
                                              layer.convDstTensorDesc, diffData,
                                              layer.convDesc, layer.convBwdFilterAlgo,
                                              workSpace, layer.convBwdFilterSizeInBytes,
                                              &vZero,
                                              layer.convFilterDesc, gconvW));
    if (layer.convBwdFilterSizeInBytes != 0)
    {
      checkCudaErrors(cudaFree(workSpace));
    }

    if (DEBUG)
      printDeviceVector(" gconvW: ", layer.w_size, gconvW);

    value_type lr = value_type(-LEARNING_RATE); // learning rate
    checkCudaErrors(cublasSaxpy(cublasHandle,
                                layer.outputs * layer.inputs * layer.kernel_dim * layer.kernel_dim,
                                &lr,
                                gconvW, 1,
                                layer.data_d, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle,
                                layer.outputs,
                                &lr,
                                gconvB, 1,
                                layer.bias_d, 1));

    if (DEBUG)
      printDeviceVector(" Updated Weights: ", layer.w_size, layer.data_d);
    if (DEBUG)
      printDeviceVector(" Updated Bias: ", layer.b_size, layer.bias_d);

    checkCudaErrors(cudaFree(gconvB));
    checkCudaErrors(cudaFree(gconvW));
    if (DEBUG)
      getchar();
  }

  void predict_example(value_type *image_data_d,
                       LAYER_NAMES_WITH_TYPE,
                       value_type *predictions,
                       int _batch_size = 1)
  {
    int n = _batch_size;
    if (DEBUG)
      println("Performing forward propagation ...");

    convoluteForward(conv1, n, image_data_d);
    convoluteForward(conv1act, n, conv1.output_d);
    convoluteForward(conv2, n, conv1act.output_d);
    convoluteForward(conv2act, n, conv2.output_d);
    poolForward(pool1, n, conv2act.output_d);

    convoluteForward(conv3, n, pool1.output_d);
    convoluteForward(conv3act, n, conv3.output_d);
    convoluteForward(conv4, n, conv3act.output_d);
    convoluteForward(conv4act, n, conv4.output_d);
    poolForward(pool2, n, conv4act.output_d);

    convoluteForward(conv5, n, pool2.output_d);
    convoluteForward(conv5act, n, conv5.output_d);
    convoluteForward(conv6, n, conv5act.output_d);
    convoluteForward(conv6act, n, conv6.output_d);
    convoluteForward(conv7, n, conv6act.output_d);
    convoluteForward(conv7act, n, conv7.output_d);
    poolForward(pool3, n, conv7act.output_d);

    convoluteForward(conv8, n, pool3.output_d);
    convoluteForward(conv8act, n, conv8.output_d);
    convoluteForward(conv9, n, conv8act.output_d);
    convoluteForward(conv9act, n, conv9.output_d);
    convoluteForward(conv10, n, conv9.output_d);
    convoluteForward(conv10act, n, conv10.output_d);
    poolForward(pool4, n, conv10act.output_d);

    convoluteForward(conv11, n, pool4.output_d);
    convoluteForward(conv11act, n, conv11.output_d);
    convoluteForward(conv12, n, conv11act.output_d);
    convoluteForward(conv12act, n, conv12.output_d);
    convoluteForward(conv13, n, conv12act.output_d);
    convoluteForward(conv13act, n, conv13.output_d);
    poolForward(pool5, n, conv13act.output_d);

    fullyConnectedForward(fc1, n, pool5.output_d);
    activationForward(fc1act, n, fc1.output_d);

    fullyConnectedForward(fc2, n, fc1act.output_d);
    activationForward(fc2act, n, fc2.output_d);

    fullyConnectedForward(fc3, n, fc2act.output_d);
    softmaxForward(fc3act, n, fc3.output_d);

    const int max_digits = fc3act.outputs;

    value_type result[n * max_digits];
    checkCudaErrors(cudaMemcpy(result, fc2act.output_d, MSIZE(n * max_digits), cudaMemcpyDeviceToHost));
    for (int batch = 0; batch < n; batch++)
    {
      predictions[batch] = 0;
      for (int i = 1; i < max_digits; i++)
      {
        if ((result[(int)predictions[batch]]) < (result[i]))
          predictions[batch] = i;
      }
    }
  }

  void learn_example(value_type *image_data_d,
                     LAYER_NAMES_WITH_TYPE,
                     value_type *targets,
                     int _batch_size = 1)
  {
    int n = _batch_size, c = fc3act.outputs;

    value_type predictions[n];

    predict_example(image_data_d, LAYER_NAMES, predictions, _batch_size);

    // if (DEBUG) println("Performing backward propagation ...");
    value_type *diffData = NULL;
    resize(n * c, &diffData);
    checkCudaErrors(cudaMemcpy(diffData, fc3act.output_d, MSIZE(n * c), cudaMemcpyDeviceToDevice));

    getDiffDataD<<<1, n>>>(targets, diffData, c, n);
    cudaDeviceSynchronize();

    value_type scalVal = 1.0f / static_cast<value_type>(n);
    checkCudaErrors(CUBLAS_SCAL(cublasHandle, n * c, &scalVal, diffData, 1));

    softmaxBackward(fc3act, n, diffData, fc3.output_d);
    fullyConnectedBackward(fc3, n, fc3act.del_d);

    activationBackward(fc2act, n, fc3.del_d, fc2.output_d);
    fullyConnectedBackward(fc2, n, fc2act.del_d);

    activationBackward(fc1act, n, fc2.del_d, fc1.output_d);
    fullyConnectedBackward(fc1, n, fc1act.del_d);

    poolBackward(pool5, n, fc1.del_d, conv13act.output_d);
    activationBackward(conv13act, n, pool5.del_d, conv13.output_d);
    convoluteBackward(conv13, n, conv13act.del_d);
    activationBackward(conv12act, n, conv13.del_d, conv12.output_d);
    convoluteBackward(conv12, n, conv12act.del_d);
    activationBackward(conv11act, n, conv12.del_d, conv11.output_d);
    convoluteBackward(conv11, n, conv11act.del_d);

    poolBackward(pool4, n, conv11.del_d, conv10act.output_d);
    activationBackward(conv10act, n, pool4.del_d, conv10.output_d);
    convoluteBackward(conv10, n, conv10act.del_d);
    activationBackward(conv9act, n, conv10.del_d, conv9.output_d);
    convoluteBackward(conv9, n, conv9act.del_d);
    activationBackward(conv8act, n, conv9.del_d, conv8.output_d);
    convoluteBackward(conv8, n, conv8act.del_d);

    poolBackward(pool3, n, conv8.del_d, conv7act.output_d);
    activationBackward(conv7act, n, pool3.del_d, conv7.output_d);
    convoluteBackward(conv7, n, conv7act.del_d);
    activationBackward(conv6act, n, conv7.del_d, conv6.output_d);
    convoluteBackward(conv6, n, conv6act.del_d);
    activationBackward(conv5act, n, conv6.del_d, conv5.output_d);
    convoluteBackward(conv5, n, conv5act.del_d);

    poolBackward(pool2, n, conv5.del_d, conv4act.output_d);
    activationBackward(conv4act, n, pool2.del_d, conv4.output_d);
    convoluteBackward(conv4, n, conv4act.del_d);
    activationBackward(conv3act, n, conv4.del_d, conv3.output_d);
    convoluteBackward(conv3, n, conv3act.del_d);

    poolBackward(pool1, n, conv3.del_d, conv2.output_d);
    activationBackward(conv2act, n, pool1.del_d, conv2.output_d);
    convoluteBackward(conv2, n, conv2act.del_d);
    activationBackward(conv1act, n, conv2.del_d, conv1.output_d);
    convoluteBackward(conv1, n, conv1act.del_d);

    // Update Weights
    fullyConnectedUpdateWeights(fc3, fc3act.del_d, fc2act.output_d, n);
    fullyConnectedUpdateWeights(fc2, fc2act.del_d, fc1act.output_d, n);
    fullyConnectedUpdateWeights(fc1, fc1act.del_d, pool5.output_d, n);

    convolutionalUpdateWeights(conv13, conv13act.del_d, conv12act.output_d);
    convolutionalUpdateWeights(conv12, conv12act.del_d, conv11act.output_d);
    convolutionalUpdateWeights(conv11, conv11act.del_d, pool4.output_d);
    convolutionalUpdateWeights(conv10, conv10.del_d, conv9act.output_d);
    convolutionalUpdateWeights(conv9, conv9act.del_d, conv8act.output_d);
    convolutionalUpdateWeights(conv8, conv8act.del_d, pool3.output_d);
    convolutionalUpdateWeights(conv7, conv7act.del_d, conv6act.output_d);
    convolutionalUpdateWeights(conv6, conv6act.del_d, conv5act.output_d);
    convolutionalUpdateWeights(conv5, conv5act.del_d, pool2.output_d);
    convolutionalUpdateWeights(conv4, conv4act.del_d, conv3act.output_d);
    convolutionalUpdateWeights(conv3, conv3act.del_d, pool1.output_d);
    convolutionalUpdateWeights(conv2, conv2act.del_d, conv1act.output_d);
    convolutionalUpdateWeights(conv1, conv1act.del_d, image_data_d);

    checkCudaErrors(cudaFree(diffData));
  }

  static void load_mnist_data(value_type **training_data, value_type **testing_data,
                              value_type **training_target, value_type **testing_target,
                              int &total_train_size, int &total_test_size)
  {
    std::string name;
    total_train_size = 0;
    total_test_size = 0;
    std::string fname;
    std::stringstream error_s;

    // Calculate total training and testing size
    for (int t = 0; t < 2; t++)
    {
      name = t == 0 ? "train" : "test";
      for (int d = 0; d < 10; d++)
      {
        std::stringstream sstm;
        sstm << "data/" << name << d << ".bin";
        fname = sstm.str();
        std::ifstream dataFile(fname.c_str(), std::ios::in | std::ios::binary);
        if (!dataFile)
        {
          error_s << "Error opening file " << fname;
          FatalError(error_s.str());
        }

        dataFile.seekg(0, std::ios::end);
        size_t file_size = static_cast<std::string::size_type>(dataFile.tellg());
        dataFile.seekg(0, std::ios::beg);
        dataFile.close();
        // println("Calculating file "<<fname<<"\t"<<file_size);
        if (t == 0)
          total_train_size += file_size;
        else
          total_test_size += file_size;
      }
    }

    *training_data = new value_type[total_train_size];
    *testing_data = new value_type[total_test_size];
    *training_target = new value_type[total_train_size / N];
    *testing_target = new value_type[total_test_size / N];
    total_train_size = 0;
    total_test_size = 0;
    for (int t = 0; t < 2; t++)
    {
      name = t == 0 ? "train" : "test";
      for (int d = 0; d < 10; d++)
      {
        std::stringstream sstm;
        sstm << "data/" << name << d << ".bin";
        fname = sstm.str();
        std::ifstream dataFile(fname.c_str(), std::ios::in | std::ios::binary);
        if (!dataFile)
        {
          error_s << "Error opening file " << fname;
          FatalError(error_s.str());
        }

        dataFile.seekg(0, std::ios::end);
        size_t file_size = static_cast<std::string::size_type>(dataFile.tellg());
        dataFile.seekg(0, std::ios::beg);

        char *data = new char[file_size];
        if (!dataFile.read(data, file_size))
        {
          error_s << "Error reading file " << fname;
          FatalError(error_s.str());
        }
        dataFile.close();

        value_type v;
        int m = file_size / N;
        // println("Reading file "<<fname<<" "<<file_size<<" "<<m);
        for (int i = 0; i < file_size; i++)
        {
          v = static_cast<value_type>((uint8_t)data[(i / N) + m * (i % N)]);
          if (t == 0)
          {
            (*training_data)[total_train_size + i] = v;
            if (i < m)
              (*training_target)[total_train_size / N + i] = d;
          }
          else
          {
            (*testing_data)[total_test_size + i] = v;
            if (i < m)
              (*testing_target)[total_test_size / N + i] = d;
          }
        }
        if (t == 0)
          total_train_size += file_size;
        else
          total_test_size += file_size;
        delete[] data;
      }
    }
  }
};

cv::Mat load_image(std::string example)
{
    cv::Mat image = cv::imread(example, cv::IMREAD_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

std::pair<int, cv::Mat *> load_all()
{
    std::string scratch = getenv("SCRATCH");
    std::string img_path = scratch + "/imagenette/imagenette2/train/*.JPEG";
    // cv::String img_path = scratch + "/imagenette/imagenette2/val/n03888257/*.JPEG";
    std::vector<cv::String> new_filename_vector;
    cv::glob(img_path, new_filename_vector, true);
    std::cout << new_filename_vector.size() << "\n";

    cv::Mat *data = new cv::Mat[new_filename_vector.size()];
    for (int i = 0; i < new_filename_vector.size(); i++)
    {
        data[i] = load_image(new_filename_vector[i]);
    }
    return {new_filename_vector.size(), data};
}

int main(int argc, char **argv)
{
  fs::path base_dir;
  fs::path output_dir;
  // Parse input
  if (find_arg_idx(argc, argv, base_dir, output_dir) < 0)
  {
    std::cout << "Usage: <program> <images/weights directory> -o[optional] <path to output directory>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "-h: see this help" << std::endl;
    std::cout << "-o <path>: path to output directory" << std::endl;
    return 0;
  }

  typedef MATRIX_DATA_TYPE value_type;
  network_t<value_type> network {};

  NETWORK_ARCH

  float *input;
  auto res = load_all();
  int len = res.first;
  len = 1;
  cv::Mat *data = res.second;
  input = (float *)malloc(len * IMAGE_SIZE);
  for (int i = 0; i < len; i++)
  {
      float *fptr = data[i].ptr<float>(0);
      std::copy(fptr, fptr + IMAGE_SIZE, input + IMAGE_SIZE * i);
  }

  float *output;
  network.learn_example(input, LAYER_NAMES, output, 1);

}