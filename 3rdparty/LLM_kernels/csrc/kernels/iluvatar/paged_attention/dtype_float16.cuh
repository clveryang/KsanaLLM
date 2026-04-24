/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * and
 * https://github.com/vllm-project/vllm/blob/v0.2.3/csrc/attention/dtype_float16.cuh
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "dtype_float32.cuh"
#include "paged_attention_generic.cuh"

#include <stdint.h>
#include <cuda_fp16.h>  // IXCC 路径用 __hadd2/__hmul2/__hfma2 替代 PTX 内联

namespace llm_kernels {
namespace iluvatar {
namespace iluvatar_fp16_detail {
// uint16_t <-> __half、uint32_t <-> __half2 的零开销 bit-cast helper。
// vLLM 原版用 PTX 内联（mul.f16x2 等），IXCC 的 ivcore 后端不认 NV PTX 指令；
// 这里改走标准 CUDA half intrinsic（IXCC 原生支持），通过 union 复用其 bit pattern。
__forceinline__ __device__ __half u16_to_h(uint16_t v) {
  union { uint16_t u; __half h; } t; t.u = v; return t.h;
}
__forceinline__ __device__ uint16_t h_to_u16(__half h) {
  union { uint16_t u; __half h; } t; t.h = h; return t.u;
}
__forceinline__ __device__ __half2 u32_to_h2(uint32_t v) {
  union { uint32_t u; __half2 h; } t; t.u = v; return t.h;
}
__forceinline__ __device__ uint32_t h2_to_u32(__half2 h) {
  union { uint32_t u; __half2 h; } t; t.h = h; return t.u;
}
}  // namespace iluvatar_fp16_detail
}  // namespace iluvatar
}  // namespace llm_kernels

namespace llm_kernels {
namespace iluvatar {

// FP16 vector types for Q, K, V.
template <>
struct Vec<uint16_t, 1> {
  using Type = uint16_t;
};
template <>
struct Vec<uint16_t, 2> {
  using Type = uint32_t;
};
template <>
struct Vec<uint16_t, 4> {
  using Type = uint2;
};
template <>
struct Vec<uint16_t, 8> {
  using Type = uint4;
};

// FP32 accumulator vector types corresponding to Vec.
template <>
struct FloatVec<uint16_t> {
  using Type = float;
};
template <>
struct FloatVec<uint32_t> {
  using Type = float2;
};
template <>
struct FloatVec<uint2> {
  using Type = Float4_;
};
template <>
struct FloatVec<uint4> {
  using Type = Float8_;
};

// Utility functions for type conversions. 全部走 CUDA half intrinsic，IXCC 友好。
inline __device__ uint32_t h0_h0(uint16_t a) {
  using namespace iluvatar_fp16_detail;
  __half h = u16_to_h(a);
  return h2_to_u32(__halves2half2(h, h));
}

inline __device__ float half_to_float(uint16_t h) {
  using namespace iluvatar_fp16_detail;
  return __half2float(u16_to_h(h));
}

inline __device__ float2 half2_to_float2(uint32_t v) {
  using namespace iluvatar_fp16_detail;
  return __half22float2(u32_to_h2(v));
}

inline __device__ uint16_t float_to_half(float f) {
  using namespace iluvatar_fp16_detail;
  return h_to_u16(__float2half_rn(f));
}

inline __device__ uint32_t float2_to_half2(float2 f) {
  using namespace iluvatar_fp16_detail;
  return h2_to_u32(__float22half2_rn(f));
}

// Vector addition.
inline __device__ uint16_t add(uint16_t a, uint16_t b) {
  using namespace iluvatar_fp16_detail;
  return h_to_u16(__hadd(u16_to_h(a), u16_to_h(b)));
}

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  using namespace iluvatar_fp16_detail;
  return h2_to_u32(__hadd2(u32_to_h2(a), u32_to_h2(b)));
}

inline __device__ uint2 add(uint2 a, uint2 b) {
  uint2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ uint4 add(uint4 a, uint4 b) {
  uint4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float2 add(uint32_t a, float2 fb) {
  float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(uint2 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add(uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

// Vector multiplication.
template <>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
  using namespace iluvatar_fp16_detail;
  return h_to_u16(__hmul(u16_to_h(a), u16_to_h(b)));
}

template <>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  using namespace iluvatar_fp16_detail;
  return h2_to_u32(__hmul2(u32_to_h2(a), u32_to_h2(b)));
}

template <>
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

template <>
inline __device__ uint2 mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  return c;
}

template <>
inline __device__ uint4 mul(uint4 a, uint4 b) {
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
  return c;
}

template <>
inline __device__ uint4 mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
  return c;
}

template <>
inline __device__ float mul(uint16_t a, uint16_t b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

template <>
inline __device__ float2 mul(uint32_t a, uint32_t b) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float2 mul(uint16_t a, uint32_t b) {
  return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ Float4_ mul(uint2 a, uint2 b) {
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  return fc;
}

template <>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint4 a, uint4 b) {
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  using namespace iluvatar_fp16_detail;
  return h2_to_u32(__hfma2(u32_to_h2(a), u32_to_h2(b), u32_to_h2(c)));
}

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) { return fma(h0_h0(a), b, c); }

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
  uint32_t s = h0_h0(a);
  uint2 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
  uint4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
  uint32_t s = h0_h0(a);
  uint4 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) { return fma(h0_h0(a), b, fc); }

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
template <>
inline __device__ float sum(uint16_t v) {
  return half_to_float(v);
}

template <>
inline __device__ float sum(uint32_t v) {
  float2 tmp = half2_to_float2(v);
  return tmp.x + tmp.y;
}

template <>
inline __device__ float sum(uint2 v) {
  uint32_t c = add(v.x, v.y);
  return sum(c);
}

template <>
inline __device__ float sum(uint4 v) {
  uint32_t c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
  return sum(c);
}

// From float32 to float16.
inline __device__ void from_float(uint16_t& dst, float src) { dst = float_to_half(src); }

inline __device__ void from_float(uint32_t& dst, float2 src) { dst = float2_to_half2(src); }

inline __device__ void from_float(uint2& dst, Float4_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
}

inline __device__ void from_float(uint4& dst, Float8_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}

// From float16 to float32.
inline __device__ float to_float(uint16_t u) { return half_to_float(u); }

inline __device__ float2 to_float(uint32_t u) { return half2_to_float2(u); }

inline __device__ Float4_ to_float(uint2 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  return tmp;
}

inline __device__ Float8_ to_float(uint4 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  tmp.z = half2_to_float2(u.z);
  tmp.w = half2_to_float2(u.w);
  return tmp;
}

// Zero-out a variable.
inline __device__ void zero(uint16_t& dst) { dst = uint16_t(0); }

}  // namespace iluvatar
}  // namespace llm_kernels
