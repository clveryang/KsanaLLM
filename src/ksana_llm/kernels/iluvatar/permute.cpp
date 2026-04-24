/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "ksana_llm/kernels/permute.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstring>

namespace ksana_llm {

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  if (permutation.size() == 2 && permutation[0] == 1 && permutation[1] == 0 && input_tensor.shape.size() == 2) {
    size_t rows = input_tensor.shape[0];
    size_t cols = input_tensor.shape[1];
    size_t elem_size = GetTypeSize(input_tensor.dtype);
    if (elem_size == 4) {
      cublasHandle_t handle;
      cublasCreate(&handle);
      cublasSetStream(handle, stream.Get());
      float alpha = 1.0f, beta = 0.0f;
      cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  rows, cols, &alpha,
                  static_cast<const float*>(input_tensor.GetPtr<void>()), cols,
                  &beta,
                  static_cast<const float*>(output_tensor.GetPtr<void>()), rows,
                  static_cast<float*>(output_tensor.GetPtr<void>()), rows);
      cublasDestroy(handle);
    } else if (elem_size == 2) {
      cudaStreamSynchronize(stream.Get());
      size_t total = rows * cols * elem_size;
      std::vector<char> h_in(total), h_out(total);
      cudaMemcpy(h_in.data(), input_tensor.GetPtr<void>(), total, cudaMemcpyDeviceToHost);
      for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
          memcpy(&h_out[(c * rows + r) * elem_size], &h_in[(r * cols + c) * elem_size], elem_size);
        }
      }
      cudaMemcpy(output_tensor.GetPtr<void>(), h_out.data(), total, cudaMemcpyHostToDevice);  // sync: avoid h_out being freed before async copy completes
    } else {
      return Status(RET_UNDEFINED_REFERENCE, "Permute: unsupported element size");
    }
    return Status();
  }
  return Status(RET_UNDEFINED_REFERENCE, "Permute: only {1,0} 2D transpose supported on Iluvatar");
}

}  // namespace ksana_llm
