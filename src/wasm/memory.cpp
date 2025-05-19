#include <stdlib.h>
#include <emscripten.h>

extern "C" {

// Generic memory allocation utilities
EMSCRIPTEN_KEEPALIVE
void* allocate_memory(size_t size) {
    void* ptr = malloc(size);
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void free_memory(void* ptr) {
    free(ptr);
}

} // extern "C" 