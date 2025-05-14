#include <stdint.h>
#include <emscripten.h>

extern "C" {

EMSCRIPTEN_KEEPALIVE
int32_t add(int32_t a, int32_t b) {
    return a + b;
}

}
