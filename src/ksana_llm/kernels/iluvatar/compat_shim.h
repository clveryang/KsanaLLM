/* Iluvatar compatibility shim — define __CUDA_ALIGN__ if not yet available.
 * NVIDIA cuda_runtime.h defines it; Iluvatar needs it from iluvatar_fp8.hpp
 * or we define it ourselves.
 * ===========================================================================*/
#pragma once

#ifndef __CUDA_ALIGN__
#  define __CUDA_ALIGN__(n) __attribute__((aligned(n)))
#endif
