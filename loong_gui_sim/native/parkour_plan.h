// SPDX-License-Identifier: MIT
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Standalone Lite26 deployment ABI. This is not the legacy Nabo Plan ABI.
// Each context is owned by one simulation thread. All errors are explicit;
// failed inference never writes a fallback or partially updated action.
unsigned loong_parkour_abi_version(void);
const char* loong_parkour_contract(void);
void* loong_parkour_create(const char* model_path, const char* ort_library_path,
                          char* error, size_t error_capacity);
int loong_parkour_infer(void* context, const float* obs, size_t obs_count,
                        const float* depth, size_t depth_count,
                        float* actions, size_t action_count,
                        char* error, size_t error_capacity);
void loong_parkour_destroy(void* context);
#ifdef __cplusplus
}
#endif
