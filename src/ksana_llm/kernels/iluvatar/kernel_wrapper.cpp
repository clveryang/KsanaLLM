/* Copyright 2026 Tencent Inc.  All rights reserved.

Iluvatar backend kernel wrappers.  At this MVP stage most ops are stubs that
return RET_UNDEFINED_REFERENCE.  Real implementations will call into ixformer
(libixformer_kernels.so) which already covers rms_norm / rotary_embedding /
paged_attention / reshape_and_cache_v4 / silu_and_mul.
==============================================================================*/
#include "ksana_llm/kernels/iluvatar/kernel_wrapper.h"

namespace ksana_llm {}  // namespace ksana_llm
