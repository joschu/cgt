#pragma once
#include <algorithm>
#include <cfloat>
// Copied & modified from Caffe


struct conv_closure {
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
};


template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <typename Dtype>
void max_pool(conv_closure* cl, cgtArray* bottom, cgtArray* top, cgtArray* mask) {
  using std::max;
  using std::min;
  Dtype* bottom_data = static_cast<Dtype*>(bottom->data());
  Dtype* top_data = static_cast<Dtype*>(top->data());
  const int top_count = top->size();
  // We'll output the mask to top[1] if it's of size >1.
  int* mask_data = static_cast<int*>(mask->data());
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  caffe_set(top_count, -1, mask_data);
    // The main loop

  int batchsize = top->shape()[0],
      channels = top->shape()[1],
      pooledheight = top->shape()[2],
      pooledwidth = top->shape()[3],
      height = bottom->shape()[2],
      width = bottom->shape()[3];

  for (int n = 0; n < batchsize; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooledheight; ++ph) {
        for (int pw = 0; pw < pooledwidth; ++pw) {
          int hstart = ph * cl->stride_h - cl->pad_h;
          int wstart = pw * cl->stride_w - cl->pad_w;
          int hend = min(hstart + cl->kernel_h, height);
          int wend = min(wstart + cl->kernel_w, width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooledwidth + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              if (bottom_data[index] > top_data[pool_index]) {
                top_data[pool_index] = bottom_data[index];
                mask_data[pool_index] = index;
              }
            }
          }
        }
      }
      bottom_data += bottom->stride(1);
      top_data += top->stride(1);
      mask_data += top->stride(1);
    }
  }
}

template <typename Dtype>
void max_pool_pullback(cgtArray* bottom, cgtArray* top, cgtArray* mask, 
  cgtArray* top_diff, cgtArray* bottom_diff) {
  const Dtype* top_diff_data = static_cast<Dtype*>(top_diff->data());
  Dtype* bottom_diff_data = static_cast<Dtype*>(bottom_diff->data());
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom_diff->size(), Dtype(0), bottom_diff_data);
  // We'll output the mask to top[1] if it's of size >1.
  int* mask_data = static_cast<int*>(mask->data());

  int batchsize = top->shape()[0],
      channels = top->shape()[1],
      pooledheight = top->shape()[2],
      pooledwidth = top->shape()[3],
      height = bottom->shape()[2],
      width = top->shape()[3];

  for (int n = 0; n < batchsize; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooledheight; ++ph) {
        for (int pw = 0; pw < pooledwidth; ++pw) {
          const int index = ph * pooledwidth + pw;
          const int bottom_index = mask_data[index];
          bottom_diff_data[bottom_index] += top_diff_data[index];
        }
      }
      bottom_diff_data += bottom->stride(1);
      top_diff_data += top->stride(1);
      mask_data += mask->stride(1);
    }
  }

}
