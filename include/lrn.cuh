// Copied from Minerva

template<typename Dtype>
__global__ static void LRNFillScale(const int nthreads, const Dtype* in, const int num, const int channels, const int height, const int width, const int size, const Dtype alpha_over_size, Dtype* scale) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x ) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    const Dtype* shifted_in = in + offset;
    Dtype* shifted_scale = scale + offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += shifted_in[head * step] * shifted_in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += shifted_in[head * step] * shifted_in[head * step];
      shifted_scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += shifted_in[head * step] * shifted_in[head * step];
      accum_scale -= shifted_in[(head - size) * step] * shifted_in[(head - size) * step];
      shifted_scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= shifted_in[(head - size) * step] * shifted_in[(head - size) * step];
      shifted_scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

template<typename Dtype>
__global__ static void LRNComputeOutput(const int nthreads, const Dtype* in,
    const Dtype* scale, const Dtype negative_beta, Dtype* out) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x ) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template<typename Dtype>
__global__ static void LRNComputeDiff(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_data, const Dtype* scale, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio,
    Dtype* bottom_diff) {

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x ) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    const Dtype* shifted_btm_data = bottom_data + offset;
    const Dtype* shifted_top_data = top_data + offset;
    const Dtype* shifted_scale = scale + offset;
    const Dtype* shifted_top_diff = top_diff + offset;
    Dtype* shifted_btm_diff = bottom_diff + offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad) {
      accum_ratio += shifted_top_diff[head * step] * shifted_top_data[head * step] /
        shifted_scale[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_ratio += shifted_top_diff[head * step] * shifted_top_data[head * step] /
        shifted_scale[head * step];
      shifted_btm_diff[(head - post_pad) * step] = shifted_top_diff[(head - post_pad) * step]
        * pow(shifted_scale[(head - post_pad) * step], negative_beta) - cache_ratio *
        shifted_btm_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += shifted_top_diff[head * step] * shifted_top_data[head * step] /
        shifted_scale[head * step];
      accum_ratio -= shifted_top_diff[(head - size) * step] *
        shifted_top_data[(head - size) * step] / shifted_scale[(head - size) * step];
      shifted_btm_diff[(head - post_pad) * step] = shifted_top_diff[(head - post_pad) * step]
        * pow(shifted_scale[(head - post_pad) * step], negative_beta) - cache_ratio *
        shifted_btm_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_ratio -= shifted_top_diff[(head - size) * step] *
        shifted_top_data[(head - size) * step] / shifted_scale[(head - size) * step];
      shifted_btm_diff[(head - post_pad) * step] = shifted_top_diff[(head - post_pad) * step]
        * pow(shifted_scale[(head - post_pad) * step], negative_beta) - cache_ratio *
        shifted_btm_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}
