
## Dependencies

- Numpy
- CUDA (Optional)

## Build

These instructions will assume that the build directory is placed in the source directory, however, you can place it anywhere. First set `PYTHONPATH` to contain the source directory as well as the `lib` subdirectory of the build directory:

    export PYTHONPATH=/path/to/cgt:/path/to/cgt/build/lib:$PYTHONPATH

Then, in the source directory, 

    mkdir build
    cd build
    cmake .. -DCGT_ENABLE_CUDA=ON # IF you have CUDA, otherwise omit this option
    make -j

## Test

In source directory, 

    python test/run_all_tests.py

