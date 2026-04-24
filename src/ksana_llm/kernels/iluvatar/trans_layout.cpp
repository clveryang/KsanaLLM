/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "ksana_llm/kernels/trans_layout.h"

namespace ksana_llm {
Status TransLayout(Tensor& tensor, Stream& stream) {
  return Status(RET_UNDEFINED_REFERENCE, "TransLayout not supported on Iluvatar yet.");
}
}  // namespace ksana_llm
