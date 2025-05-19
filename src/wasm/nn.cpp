#include <stdint.h>
#include <emscripten.h>

// Define the struct
struct __attribute__((packed)) Point {
    int32_t x;
    int32_t y;
};

extern "C" {

// Function to add two points, result stored in an output parameter
EMSCRIPTEN_KEEPALIVE
void add_points(Point* p1, Point* p2, Point* result_out) {
    if (!p1 || !p2 || !result_out) {
        if (result_out) {
            result_out->x = 0;
            result_out->y = 0;
        }
        return;
    }
    result_out->x = p1->x + p2->x;
    result_out->y = p1->y + p2->y;
}

} // extern "C"
