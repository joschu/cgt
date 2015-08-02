
# Installation

Dependencies

- NumPy
- Cython (Optional)
- CUDA (Optional)



## Option 1: pure python only

No compilation is required for this option, and NumPy is the only dependency.
Just update your `PYTHONPATH` as follows:

    export PYTHONPATH=/path/to/cgt

## Option 2: build C++ backend

If you want to use the C++ backend, which has better performance and in the future will allow multithreading, then Cython is required, and the installation procedure is as follows.
First, `cd` into the source directory. Then, type

    mkdir build
    cd build
    cmake ..
    make -j

Then add two directories to your `PYTHONPATH`

    export PYTHONPATH=/path/to/cgt:/path/to/cgt/build/lib:$PYTHONPATH


Note: if you don't want to put the `build` directory inside your source directory, you can put it in some other location, just make sure to alter alter your `PYTHONPATH` accordingly.

## Option 3: build C++ backend with CUDA

Follow the instructons for Option 2, but alter the cmake command to instead read

    cmake .. -DCGT_ENABLE_CUDA=ON

# Running unit tests

In source directory, 

    python test/run_all_tests.py

