CXX = emcc
CXXFLAGS = -Os -Wall -Wextra -Wswitch-enum
LDFLAGS = -sSIDE_MODULE=1

INCLUDES = -Isrc/core
SRC = src/wasm/bindings.cpp

BUILD_DIR = web/public
TARGET = $(BUILD_DIR)/output.wasm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)