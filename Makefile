CXX = emcc
CXXFLAGS = -O3 -Wall -Wextra -Wswitch-enum --no-entry
LDFLAGS = -sALLOW_MEMORY_GROWTH=1 -sEXPORT_ALL=1 -Wl,--allow-undefined

INCLUDES = -Isrc/core -Isrc/wasm
SRC = src/wasm/memory.cpp src/wasm/nn.cpp src/core/nn.cpp

BUILD_DIR = web/public
TARGET = $(BUILD_DIR)/output.wasm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
