# Neural Network from Scratch (C++ to WASM)

## Overview

In this project, I aim to build a neural network library from scratch using C++,
and compile it to WebAssembly (WASM). The library will then be used in a web
interface.

## Technologies

- C++
- Emscripten
- WebAssembly (WASM)
- Astro

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Emscripten SDK**: This project uses Emscripten to compile C++ to
  WebAssembly. Make sure you have the Emscripten SDK installed and the `emcc`
  compiler is in your system's PATH. You can find installation instructions on
  the
  [Emscripten website](https://emscripten.org/docs/getting_started/downloads.html).

## Getting Started

### Build the WebAssembly module

To build the WebAssembly module, navigate to the project's root directory in
your terminal and run:

```bash
make
```

This command will compile the C++ source code located in `src/wasm/bindings.cpp`
and output the WebAssembly module to `web/public/output.wasm`.

To clean the build artifacts, you can run:

```bash
make clean
```

This will remove the `web/public/output.wasm` file.

### Run the web application

Refer to the [web/README.md](web/README.md) file for instructions on how to run
the web application.
