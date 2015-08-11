#pragma once
#include "cgt_common.h"
#include "cudnn.h"
#include "cgt_cuda.h"

#define CUDNN_CHECK(status) \
  do { \
    if (status != CUDNN_STATUS_SUCCESS) {puts(cudnnGetErrorString(status)); abort();} \
  } while (0)

const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}

struct conv_closure {
  int pad_height, pad_width, stride_vertical, stride_horizontal;
  cudnnHandle_t handle;
  cudaStream_t stream;
};


cudnnDataType_t cudnnDataType(cgtDtype d) {
  if (d == cgt_i4) return CUDNN_DATA_FLOAT;
  else if (d==cgt_i8) return CUDNN_DATA_DOUBLE;
  else assert(0 && "Invalid datatype");
}

cudnnConvolutionDescriptor_t createConvDescr(conv_closure* cl) {
  cudnnConvolutionDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc, cl->pad_height,
   cl->pad_width, cl->stride_vertical, cl->stride_horizontal, 1, 1, CUDNN_CONVOLUTION));
  return desc;
}

cudnnTensorDescriptor_t createTensorDescr(cgtArray* a) {
  cudnnTensorDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  assert(a->ndim() > 0 && a->ndim() <= 4);
  assert(a->dtype() == cgt_f4);
  int strides[4];
  int shape[4];
  for (int i=0; i < a->ndim(); ++i) {
    strides[i] = a->stride(i);
    shape[i] = a->shape()[i];
  }
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, a->ndim(), shape, strides));
}

cudnnFilterDescriptor_t createFilterDescr(cgtArray* a) {
  cudnnFilterDescriptor_t desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
  assert(a->ndim() > 0 && a->ndim() <= 4);
  assert(a->dtype() == cgt_f4);
  int shape[4];
  for (int i=0; i < a->ndim(); ++i) shape[i] = a->shape()[i];
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(&desc, CUDNN_DATA_FLOAT, a->ndim(), shape));  
}


void destroyConvDescr(cudnnConvolutionDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(&desc));
}

void destroyTensorDescr(cudnnTensorDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(&desc));  
}

void destroyFilterDescr(cudnnFilterDescriptor_t desc) {
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc));  
}

extern "C" void setupConv(conv_closure* cl) {
  CUDA_CHECK(cudaStreamCreate(cl->stream));
  CUDNN_CHECK(cudnnCreate(cl->handle));
  CUDNN_CHECK(cudnnSetStream(cl->handle, cl->stream));
}

extern "C" void teardownConv(conv_closure* cl) {
  CUDNN_CHECK(cudnnDestroy(cl->handle));
  CUDA_CHECKcudaStreamDestroy(cl->stream));
}

void performConvForward(conv_closure* cl, cgtArray* bottom, cgtArray* filter, cgtArray* bias, 
  cgtArray* top) {

  auto bottom_desc = createTensorDescr(bottom);  
  auto filter_desc = createFilterDescr(filter);  
  auto bias_desc = createTensorDescr(bias);  
  auto top_desc = createTensorDescr(top);  
  auto conv_desc = createConvDescr(cl);

  float one = 1;
  float zero = 0;
  cudnnConvolutionFwdAlgo_t algorithm;
  size_t workspace_size;
  void* workspace;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cl->handle, bottom_desc, filter_desc, conv_desc, top_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm));
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cl->handle, bottom_desc, filter_desc, conv_desc, top_desc, algorithm, &workspace_size));
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  CUDNN_CHECK(cudnnConvolutionForward(cl->handle, &one, bottom_desc, bottom->data(), filter_desc, filter->data(), conv_desc, algorithm, workspace, workspace_size, &zero, top_desc, top));
  CUDNN_CHECK(cudnnAddTensor(cl->handle, CUDNN_ADD_SAME_C, &one, bias_desc, bias, &one, top_desc, top));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction
  CUDA_CHECK(cudaFree(workspace));

  destroyConvDescr(conv_desc);
  destoryTensorDescr(top_desc);
  destoryTensorDescr(bias_desc);
  destroyFilterDescr(filter_desc);
  destoryTensorDescr(bottom_desc);
}

void performConvBackwardData(conv_closure* cl, cgtArray* top_diff, cgtArray* filter, cgtArray* bottom_diff) {

  auto top_diff_desc = createTensorDescr(top_diff);  
  auto filter_desc = createFilterDescr(filter);  
  auto bottom_desc = createTensorDescr(bottom_diff);  
  auto conv_desc = createConvDescr(cl);

  float one = 1;
  float zero = 0;
  CUDNN_CHECK(cudnnConvolutionBackwardData(cl->handle, &one, filter_desc, filter->data(), top_diff_desc, top_diff->data(), conv_desc, &zero, bottom_diff_desc, bottom_diff->data()));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyConvDescr(conv_desc)
  destroyTensorDescr(bottom_diff);
  destroyFilterDescr(filter);
  destroyTensorDescr(top_diff);

}

void performConvBackwardFilter(conv_closure* cl, cgtArray* top_diff, cgtArray* bottom, cgtArray* filter_diff) {

  auto bottom_desc = createTensorDescr(bottom_desc);  
  auto filter_desc = createFilterDescr(filter_diff_desc);  
  auto top_desc = createTensorDescr(top_diff_desc);  
  auto conv_desc = createConvDescr(cl);

  float one = 1;
  float zero = 0;
  CUDNN_CHECK(cudnnConvolutionBackwardFilter(cl->handle, &one, bottom_desc, bottom->data(), top_diff_desc, top_diff->data(), conv_desc, &zero, filter_diff_desc, filter_diff->data));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyTensorDescr(conv_desc);
  destroyTensorDescr(top_diff_desc);
  destroyFilterDescr(filter_diff_desc);
  destroyTensorDescr(bottom_desc);
}

void performConvBackwardBias(conv_closure* cl, cgtArray* top_diff, cgtArray* bias_diff) {

  auto top_diff_desc = createTensorDescr(top_diff);
  auto bias_diff_desc = createTensorDescr(bias_diff);

  float one = 1;
  float zero = 0;
  CUDNN_CHECK(cudnnConvolutionBackwardBias(cl->handle, &one, top_diff_desc, top_diff->data(), &zero, bias_diff_desc, bias_diff->data));
  CUDA_CHECK(cudaStreamSynchronize(cl->stream));  // Synchronize before destruction

  destroyTensorDescr(bias_diff_desc);
  destroyTensorDescr(top_diff_desc);
}