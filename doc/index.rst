*************************
Computation Graph Toolkit
*************************

Computation Graph Toolkit (CGT) is a library for evaluation and differentiation of functions of multidimensional arrays.


.. CAUTION::

    **WORK IN PROGRESS!** This software is not yet released. There are some bugs and some rough edges. The user API may change, and the internals will change rapidly.


What Does It Do?
================

The core features are as follows:

- Automatic differentiation of functions involving multidimensional arrays, using computation graphs
- Compile fast implementations of array computations that can be run on CPU, GPU, and multi-machine configurations.
- A graph simplification system, which readily handles extremely large (1M node) graphs.
- Supports both forward and backward derivative propagation, as well as higher-order differentiation.
- Serialization/deserialization of computation graphs
- CGT can export standalone C++ code for of your functions.

CGT is motivated by large-scale machine learning and AI problems, however, the core library will focus on the more abstract problems of evaluating and differentiating mathematical expressions. This will ensure that CGT is flexible enough to handle use-cases that are completely unanticipated by the software authors. Libraries for numerical optimization and convenient construction of neural networks will be built on top of CGT’s core functionality.

With regard to previous work, CGT is most similar to Theano.
However, CGT makes some core changes that necessitated a new codebase:

1. C++/CUDA implementations of ops don't use the Python C-API. Furthermore, the computation graph is compiled into a data-structure that can be executed fully independently of python. Hence, multithreaded execution is possible.
2. Internally, CGT substantially revises the internal datastructures and compilation pipeline. The following practical benefits result:

- Much faster graph simplification, which is designed carefully to take linear time in the size of the graph.
- Most shape errors can be detected at graph construction time.

CGT aims to make it easy to to construct large and complicated models, while ensuring that the resulting code is concise and closely resembles the underlying mathematical expressions.

Installation
============

**Dependencies**:

- NumPy
- Cython (Optional)
- CUDA (Optional)



Option 1: Python (NumPy) only
-----------------------------

No compilation is required for this option, and NumPy is the only dependency.
Just update your ``PYTHONPATH`` as follows::

    export PYTHONPATH=/path/to/cgt

Option 2: Build C++ backend
---------------------------

If you want to use the C++ backend, which has better performance and in the future will allow multithreading, then Cython is required, and the installation procedure is as follows.
First, ``cd`` into the source directory. Then, type::

    mkdir build
    cd build
    cmake ..
    make -j

Then add two directories to your ``PYTHONPATH``::

    export PYTHONPATH=/path/to/cgt:/path/to/cgt/build/lib:$PYTHONPATH


Note: if you don't want to put the ``build`` directory inside your source directory, you can put it in some other location, just make sure to alter alter your ``PYTHONPATH`` accordingly.

Option 3: Build C++ backend with CUDA
-------------------------------------

Follow the instructons for Option 2, but alter the cmake command to instead read::

    cmake .. -DCGT_ENABLE_CUDA=ON

Option 4: VM with preinstalled CGT + CUDA
-----------------------------------------

TODO


Running unit tests
------------------

You can run our suite of unit tests to verify your installation. In the source directory::

    python test/run_all_tests.py

API Tour
========

.. notebook:: ../examples/api_tour.ipynb


Internals Tour
==============

.. notebook:: ../examples/internals_tour.ipynb

Debugging
=========



Cookbook
========

- ``examples/mnist_classification.py``
- ``examples/demo_char_rnn.py``
- ``examples/neural_turing_machine.py``

Links and Further Reading
=========================

