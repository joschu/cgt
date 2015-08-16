*************************
Computation Graph Toolkit
*************************

Computation Graph Toolkit (CGT) is a library for evaluation and differentiation of functions of multidimensional arrays.


.. CAUTION::

    **WORK IN PROGRESS!** This software is not ready to release! The ends are loose, the edges are VERY rough, and most of the functionality required to make this software useful are not yet implemented.

What Does It Do?
================

The core features are as follows:

- Automatic differentiation of functions involving multidimensional arrays, using computation graphs
- Compile fast implementations of array computations that can be run in parallel on multiple CPUs and GPUs. [ONLY SINGLE-THREADED CPU USAGE IS CURRENTLY SUPPORTED.]
- A compilation process that simplifies your function through arithmetic identities and in-place optimizations, which readily handles extremely large (1M operation) graphs.
- Supports both forward and backward derivative propagation, as well as higher-order differentiation.
- CGT can export standalone C++ code for of your functions. [PROSPECTIVE]

CGT is motivated by large-scale machine learning and AI problems, however, the core library will focus on the more abstract problems of evaluating and differentiating mathematical expressions. This will ensure that CGT is flexible enough to handle use-cases that are completely unanticipated by the software authors. Libraries for numerical optimization and convenient construction of neural networks will be built on top of CGT’s core functionality.

With regard to previous work, CGT is most similar to Theano.
In fact, CGT aims to mostly replicate Theano's API.
However, CGT makes some core architectural changes that necessitated a new codebase:

1. C++/CUDA implementations of ops don't use the Python C-API. Furthermore, the computation graph is compiled into a data-structure that can be executed by C++ code independently of python. Hence, multithreaded execution is possible.
2. Internally, CGT substantially reimagines the data-structures and compilation pipeline, which (in our view) leads to a cleaner codebase and makes ultra-fast compilation possible.

See :ref:`whynottheano` for a more thorough explanation of why we decided to reimplement Theano's functionality from scratch.


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

Follow the instructions for Option 2, but alter the cmake command to instead read::

    cmake .. -DCGT_ENABLE_CUDA=ON

Option 4: VM with preinstalled CGT + CUDA
-----------------------------------------

TODO


Running unit tests
------------------

You can run our suite of unit tests to verify your installation. In the source directory::

    nosetests

Note that you'll have to install the ``nose`` python module.

Tutorial
========

.. notebook:: ../examples/tutorial.ipynb


Tour of the Internals
=====================

.. notebook:: ../examples/internals_tour.ipynb

Debugging
=========



Cookbook
========

See ``examples`` directory.

Links and Further Reading
=========================



.. _whynottheano:

Why not Build on Theano?
========================

CGT is heavily based on Theano, and we (the authors of CGT) think that Theano is a beautiful and highly innovative piece of software.
However, several limitation of Theano (in its current state) motivated us to consider creating a new library:

- Optimization and compilation of the graphs is very slow. For this region, Theano becomes  inconvenient when working with large recurrent models. To use these models, one has to use the Scan operator, which is usually less convenient than constructing a graph with the unrolled computation. |br| **CGT solution**: (1) the main graph simplification process in CGT involves a single pass through the graph that applies several different types of replacement simultaneously (common subexpression elimination, constant propagation, arithmetic identities like ``x*1=x``.) In-place optimizations are performed in a second phase that also involves a single pass through the graph. Together, these phases take negligible time. Furthermore, we use a different graph data-structure (similar to SSA representations used by compilers) which allows for much cleaner simplification code. In Theano, the C++/CUDA compilation itself takes significant time, because Theano compiles a whole Python module (written in C++) for each function, which includes Python.h and numpy/arrayobject.h. On the other hand, CGT compiles a small C++ file with minimal header dependencies, taking a small fraction of a second, and the relevant function is later retrieved with ``dlopen`` and ``dlsym``.
- Theano can't straightforwardly be used to perform different operations in parallel, because of Python's GIL. |br| **CGT solution**: we create a representation of the computation called the execution graph, which can be executed in C++ independently of Python, and encodes all of the information necessary for concurrent execution of operations.
- When using GPUs, the user often obtains poor performance unless he is careful to set up the graph in a way that the operations can be executed on the GPU. |br| **CGT solution**: we give the user finer grained control over which operation is performed on which device.
- Automatic upcasting rules (e.g. int * float = double) require the user to add casts many casting operations. |br| **CGT solution**: we globally choose either single or double (or quad) precision, using ``cgt.set_precision(...)``
- It is difficult to debug certain bugs problems such as shape mismatches. Furthermore, Theano tensors have a `broadcastable` attribute that must be set to allow broadcasting and is point of confusion for many users. |br| **CGT solution**: we require explicit broadcasting using the ``broadcast(...)`` function. This requires slightly more verbosity but serves to eliminate many common errors and usually allows us to determine all of the shapes of intermediate variables in terms of the shapes of the inputs, which allows many shape errors to be caught at graph construction time.

Some of issues could be addressed within Theano's existing codebase, however, we believe that by breaking compatibility and starting from afresh, it will be possible to resolve them more cleanly.


.. |br| raw:: html

   <br />
