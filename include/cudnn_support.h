#pragma once
#include "cgt_common.h"
#include "cudnn.h"
#include "cgt_cuda.h"


#define CUDNN_CHECK(status) \
  do { \
    if (status != CUDNN_STATUS_SUCCESS) {printf("%s (%s:%d)\n", cudnnGetErrorString(status), __FILE__, __LINE__); abort();} \
  } while (0)

struct conv_closure {
  int pad_height, pad_width, stride_vertical, stride_horizontal;
  cudnnHandle_t handle;
  cudaStream_t stream;
};


struct pooling_closure {
  int kernel_height, kernel_width, 
      pad_height, pad_width, 
      stride_vertical, stride_horizontal;
  cudnnHandle_t handle;
  cudaStream_t stream;
};

union FloatOrDouble {
  float f;
  double d;
};
static inline FloatOrDouble value2union(double x, cgtDtype dtype) {
  FloatOrDouble out;
  if (dtype == cgt_f4) out.f=x;
  else if (dtype == cgt_f8) out.d=x;
  else cgt_assert(0 && "Invalid datatype");
  return out;
}


cudnnDataType_t cudnnDataType(cgtDtype d) {
  if (d == cgt_f4) return CUDNN_DATA_FLOAT;
  else if (d==cgt_f8) return CUDNN_DATA_DOUBLE;
  else {
    printf("invalid datatype %i", d);
    abort();
  }
}

cudnnConvolutionDescriptor_t createConvDescr(conv_closure* cl) {
  int padA[2] = {cl->pad_height, cl->pad_width};
  int strideA[2] = {cl->stride_vertical, cl->stride_horizontal};
  int upscaleA[2] = {1, 1};
  int ndims = 2;
  cudnnConvolutionDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(desc, 
    ndims,
    padA,
    strideA,
    upscaleA,    
    CUDNN_CONVOLUTION));
  return desc;
}

cudnnTensorDescriptor_t createTensorDescr(cgtArray* a) {
  cudnnTensorDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  cgt_assert(a->ndim() > 0 && a->ndim() <= 4);
  int strides[4];
  int shape[4];
  for (int i=0; i < a->ndim(); ++i) {
    strides[i] = a->stride(i);
    shape[i] = a->shape()[i];
  }
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, cudnnDataType(a->dtype()), a->ndim(), shape, strides));
  return desc;
}

cudnnFilterDescriptor_t createFilterDescr(cgtArray* a) {
  cudnnFilterDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
  assert(a->ndim() > 0 && a->ndim() <= 4);
  int shape[4];
  for (int i=0; i < a->ndim(); ++i) shape[i] = a->shape()[i];
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(desc, cudnnDataType(a->dtype()), a->ndim(), shape));
  return desc;
}

cudnnPoolingDescriptor_t createPoolingDescr(pooling_closure* pc) {
  cudnnPoolingDescriptor_t desc;
  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  int ndims=2;
  int windowDimA[2] = {pc->kernel_height, pc->kernel_width};
  int paddingA[2] = {pc->pad_height, pc->pad_width};
  int strideA[2] = {pc->stride_vertical, pc->stride_horizontal};
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(desc, 
    mode,
    ndims,
    windowDimA,
    paddingA,
    strideA));
  return desc;  
}

void destroyConvDescr(cudnnConvolutionDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(desc));
}

void destroyTensorDescr(cudnnTensorDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));  
}

void destroyFilterDescr(cudnnFilterDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc));  
}

void destroyPoolingDescr(cudnnPoolingDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc));
}

template <typename ClType>
void setup_cudnn(ClType* cl) {
  CUDA_CHECK(cudaStreamCreate(&cl->stream));
  CUDNN_CHECK(cudnnCreate(&cl->handle));
  CUDNN_CHECK(cudnnSetStream(cl->handle, cl->stream));
}

template <typename ClType>
void teardown_cudnn(ClType* cl) {
  CUDNN_CHECK(cudnnDestroy(cl->handle));
  CUDA_CHECK(cudaStreamDestroy(cl->stream));
}

void performConvForward(conv_closure* cl, cgtArray* bottom, cgtArray* filter, cgtArray* bias, 
  cgtArray* top) {

  cgt_assert(bottom->devtype() == cgtGPU && top->devtype() == cgtGPU && filter->devtype() == cgtGPU && bias->devtype() == cgtGPU);

  auto bottom_desc = createTensorDescr(bottom);  
  auto filter_desc = createFilterDescr(filter);
  auto bias_desc = createTensorDescr(bias);  
  auto top_desc = createTensorDescr(top);  
  auto conv_desc = createConvDescr(cl);


  int outputdim[4] = {0,0,0,0};
  CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(conv_desc, bottom_desc, filter_desc, 4, outputdim));
  for (int i=0; i < top->ndim(); ++i) cgt_assert(outputdim[i]==top->shape()[i]);

  FloatOrDouble zero=value2union(0, bottom->dtype()), one = value2union(1, bottom->dtype());
  cudnnConvolutionFwdAlgo_t algorithm;
  size_t workspace_size;
  void* workspace;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cl->handle, bottom_desc, filter_desc, conv_desc, top_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm));
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cl->handle, bottom_desc, filter_desc, conv_desc, top_desc, algorithm, &workspace_size));
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  CUDNN_CHECK(cudnnConvolutionForward(cl->handle, &one, bottom_desc, bottom->data(), filter_desc, filter->data(), conv_desc, algorithm, workspace, workspace_size, &zero, top_desc, top->data()));
  CUDNN_CHECK(cudnnAddTensor(cl->handle, CUDNN_ADD_SAME_C, &one, bias_desc, bias->data(), &one, top_desc, top->data()));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction
  CUDA_CHECK(cudaFree(workspace));

  destroyConvDescr(conv_desc);
  destroyTensorDescr(top_desc);
  destroyTensorDescr(bias_desc);
  destroyFilterDescr(filter_desc);
  destroyTensorDescr(bottom_desc);
}

void performConvBackwardData(conv_closure* cl, cgtArray* top_diff, cgtArray* filter, cgtArray* bottom_diff) {

  auto top_diff_desc = createTensorDescr(top_diff);  
  auto filter_desc = createFilterDescr(filter);  
  auto bottom_diff_desc = createTensorDescr(bottom_diff);  
  auto conv_desc = createConvDescr(cl);

  FloatOrDouble zero=value2union(0, top_diff->dtype()), one = value2union(1, top_diff->dtype());
  CUDNN_CHECK(cudnnConvolutionBackwardData(cl->handle, &one, filter_desc, filter->data(), top_diff_desc, top_diff->data(), conv_desc, &zero, bottom_diff_desc, bottom_diff->data()));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyConvDescr(conv_desc);
  destroyTensorDescr(bottom_diff_desc);
  destroyFilterDescr(filter_desc);
  destroyTensorDescr(top_diff_desc);

}

void performConvBackwardFilter(conv_closure* cl, cgtArray* top_diff, cgtArray* bottom, cgtArray* filter_diff) {

  auto bottom_desc = createTensorDescr(bottom);  
  auto filter_diff_desc = createFilterDescr(filter_diff);  
  auto top_diff_desc = createTensorDescr(top_diff);  
  auto conv_desc = createConvDescr(cl);

  FloatOrDouble zero=value2union(0, top_diff->dtype()), one = value2union(1, top_diff->dtype());
  CUDNN_CHECK(cudnnConvolutionBackwardFilter(cl->handle, &one, bottom_desc, bottom->data(), top_diff_desc, top_diff->data(), conv_desc, &zero, filter_diff_desc, filter_diff->data()));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyConvDescr(conv_desc);
  destroyTensorDescr(top_diff_desc);
  destroyFilterDescr(filter_diff_desc);
  destroyTensorDescr(bottom_desc);
}

void performConvBackwardBias(conv_closure* cl, cgtArray* top_diff, cgtArray* bias_diff) {

  auto top_diff_desc = createTensorDescr(top_diff);
  auto bias_diff_desc = createTensorDescr(bias_diff);

  FloatOrDouble zero=value2union(0, top_diff->dtype()), one = value2union(1, top_diff->dtype());
  CUDNN_CHECK(cudnnConvolutionBackwardBias(cl->handle, &one, top_diff_desc, top_diff->data(), &zero, bias_diff_desc, bias_diff->data()));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyTensorDescr(bias_diff_desc);
  destroyTensorDescr(top_diff_desc);
}

void performPoolingForward(pooling_closure* cl, cgtArray* bottom, cgtArray* top) {
  cgt_assert(bottom->devtype() == cgtGPU && top->devtype() == cgtGPU);

  auto bottom_desc = createTensorDescr(bottom);  
  auto top_desc = createTensorDescr(top);  
  auto pooling_desc = createPoolingDescr(cl);

  // int outputdim[4] = {0,0,0,0};
  // CUDNN_CHECK(cudnnGetPoolingNdForwardOutputDim(pooling_desc, bottom_desc, 4, outputdim));
  // for (int i=0; i < top->ndim(); ++i) cgt_assert(outputdim[i]==top->shape()[i]);

  FloatOrDouble zero=value2union(0, bottom->dtype()), one = value2union(1, bottom->dtype());
  CUDNN_CHECK(cudnnPoolingForward(cl->handle, pooling_desc, &one, bottom_desc, bottom->data(), &zero, top_desc, top->data()));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyPoolingDescr(pooling_desc);
  destroyTensorDescr(top_desc);
  destroyTensorDescr(bottom_desc);  
}

void performPoolingBackward(pooling_closure* cl, cgtArray* bottom, cgtArray* top, cgtArray* top_diff, cgtArray* bottom_diff) {
  cgt_assert(bottom->devtype() == cgtGPU && top->devtype() == cgtGPU && top_diff->devtype() == cgtGPU && bottom_diff->devtype() == cgtGPU);

  auto bottom_desc = createTensorDescr(bottom);  
  auto top_desc = createTensorDescr(top);  
  auto top_diff_desc = createTensorDescr(top_diff);
  auto bottom_diff_desc = createTensorDescr(bottom_diff);
  auto pooling_desc = createPoolingDescr(cl);

  FloatOrDouble zero=value2union(0, bottom->dtype()), one = value2union(1, bottom->dtype());
  CUDNN_CHECK(cudnnPoolingBackward(cl->handle, pooling_desc, &one, 
    top_desc, top->data(), 
    top_diff_desc, top_diff->data(),
    bottom_desc, bottom->data(), 
    &zero, 
    bottom_diff_desc, bottom_diff->data()
  ));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyTensorDescr(bottom_desc);
  destroyTensorDescr(top_desc);
  destroyTensorDescr(top_diff_desc); 
  destroyTensorDescr(bottom_diff_desc);
  destroyPoolingDescr(pooling_desc);
}

