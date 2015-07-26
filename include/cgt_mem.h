#pragma once
#include "cgt_utils.h"

void* cgt_alloc(char devtype, size_t size);
void cgt_free(char devtype, void* ptr);
void cgt_memcpy(char dest_type, char src_type, void* dest_ptr, void* src_ptr, size_t nbytes);
