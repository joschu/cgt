
// JDS: Modified Caffe code so channels are last dimension in output, and so arguments are template parameters
// (actually gives a good speedup). could be further sped up by rearranging loops
// so inner loops can be unrolled

template <typename Dtype, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w>
void im2col_cpu(const Dtype* data_im, int channels, int height, int width, Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[channels_col*width_col*h + channels_col*w + c] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[channels_col*width_col*h + channels_col*w + c] = 0;
      }
    }
  }
}

template <typename Dtype, int patch_h, int patch_w, int pad_h, int pad_w, int stride_h, int stride_w>
void col2im_cpu(const Dtype* data_col, const int channels, int height, int width, Dtype* data_im) {
  for (int i=0; i < height * width * channels; ++i) data_im[i] = Dtype(0);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[channels_col*width_col*h + channels_col*w + c];
      }
    }
  }
}
