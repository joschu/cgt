*************************
Computation Graph Toolkit
*************************

Computation Graph Toolkit (CGT) is a library for evaluation and differentiation of functions of multidimensional arrays.
Source code is available on `GitHub <https://github.com/joschu/cgt>`_.

What Does It Do?
================

The core features are as follows:

- Automatic differentiation of functions involving multidimensional arrays, using computation graphs
- Compile fast implementations of array computations that can be run in parallel on multiple CPUs and GPUs. (GPU and multi-GPU support is currently in work-in-progress)
- A compilation process that simplifies your function through arithmetic identities and in-place optimizations, which readily handles extremely large graphs.
.. - Supports both forward and backward derivative propagation, as well as higher-order differentiation.
.. - CGT can export standalone C++ code for of your functions.

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

- NumPy >= 1.9
- Cython >= 0.22 (optional, required for native backend)
- CUDA Tookit (optional, required for GPU implementation of Ops)


Option 1: Python (NumPy) only
-----------------------------

No compilation is required for this option, and NumPy is the only dependency.
Just update your ``PYTHONPATH`` as follows::

    export PYTHONPATH=/path/to/cgt

Option 2: Build C++ backend
---------------------------

If you want to use the C++ backend, which has better performance and enables multithreading, then Cython is required, and the installation procedure is as follows.
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

    nosetests -v

Note that you'll have to install the ``nose`` python module.

Tutorial
========

.. notebook:: ../examples/tutorial.ipynb

Configuration
=============

CGT uses a some global configuration options that you should be aware of.
You can set these options in the file ``~/.cgtrc``; see ``cgtrc.example`` in the source directory for a template.
You can also modify these values via the command line, e.g. ``CGT_FLAGS=backend=native,precision=double``.
The file, ``cgtrc_spec.ini``, included below, provides a listing of the configuration variables.

.. literalinclude:: ../cgtrc_spec.ini
    :lines: 5-

For best performance, set ``backend=native, but for development you should set ``backend=python`` because the error handling is better.

Guide to Porting Theano Code to CGT
===================================

CGT mostly replicates Theano's API, but there are a few gotchas.

- CGT does not allow automatic broadcasting of singleton dimensions (for elementwise binary operations like addition and multiplication.) Use the ``broadcast`` function::
  
    def broadcast(opname, a, b, bcpat):
        """
        Perform elementwise binary operation such as addition or multiplication, and expand
        singleton dimensions when appropriate.

        opname: string name of operation: *,+,-,/,<,>,<=,>=,**,==,!=
        a, b: variables
        bcpat: a string of x,1 specifying which dimensions are singletons in both a and b. Here are some examples:
            "x1,1x":        a.shape[1] == 1 and b.shape[0] == 1
            "xx1,xxx":      a.shape[2] == 1, but we should have a.shape[0]==b.shape[0] and a.shape[1]==b.shape[1]

        E.g., here's an example of using this function
        a = np.zeros((2,3))
        b = np.zeros((2,1))
        z = cgt.broadcast("+", a, b, "xx,x1")
        """  
        ...


nn: Neural Network Module
=========================

The ``nn`` module (``import cgt.nn``) provides a light wrapper around CGT's core API that allows the user to concisely build up complicated neural network models.
Below we will show how to build up a convolutional neural network, and then how to parallelize it using the multi-threaded interpreter.
A complete code listing can be found in ``examples/cgt_theano_feedforward_comparison.py``.


A Simple ConvNet
----------------

The following code constructs a simple convolution neural network (sized for MNIST images, what else?),
where the loss function is the negative log-likelihood of labels.
(A full listing is provided in the source directory, see ``examples/cgt_theano_feedforward_comparison.py``)

.. code-block:: python

    # X: a symbolic variable representing a batch of input images,
    # with shape (batchsize, nchannels, nrows, ncols)
    X = cgt.tensor4("X", fixed_shape=(None,1,28,28)) 
    # We provide the fixed_shape argument so 
    # CGT can infer the shape of downstream variables
    # y: a symbolic variable representing the labels, which are integers
    y = cgt.vector("y", dtype='i8')
    # SpatialConvolution(...) is a constructor call, which builds the weights 
    # (filter) and the biases for the convolutional layer
    # rectify(...) is just a function call that maps x -> x*(x>0)
    conv1 = nn.rectify(
        nn.SpatialConvolution(1, 32, kernelshape=(3,3), pad=(0,0), 
        weight_init=nn.IIDGaussian(std=.1))(X))
    # Apply max pooling function
    pool1 = nn.max_pool_2d(conv1, kernelshape=(3,3), stride=(2,2))
    # Another convolutional layer
    conv2 = nn.rectify(
        nn.SpatialConvolution(32, 32, kernelshape=(3,3), pad=(0,0), 
        weight_init=nn.IIDGaussian(std=.1))(pool1))
    pool2 = nn.max_pool_2d(conv2, kernelshape=(3,3), stride=(2,2))
    # Now we flatten the last output image
    d0,d1,d2,d3 = pool2.shape
    flatlayer = pool2.reshape([d0,d1*d2*d3])
    # CGT can infer the shape of variables
    nfeats = cgt.infer_shape(flatlayer)[1]
    # One final fully-connected layer, and then a log-softmax
    logprobs = nn.logsoftmax(nn.Affine(nfeats, 10)(flatlayer))
    neglogliks = -logprobs[cgt.arange(X.shape[0]), y]
    loss = neglogliks.mean()



Now that we've built up an expression for the loss, we can build an expression for the gradient

.. code-block:: python

    # walk through the graph and find all parameters 
    # i.e., variables constructed with cgt.shared(...)
    params = nn.get_parameters(loss)
    gparams = cgt.grad(loss, params)

Finally, we can build up a function that updates the parameters:

.. code-block:: python

    updates = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
    updater = cgt.function([X, y, stepsize], loss, updates=updates)


Adding parallelization
----------------------

CGT is capable of executing computations in parallel.
A feedforward network does not offer much opportunity for parallelization, but we can easily transform it to use data parallelism.

First let's build a ``Module``, which is a parameterized function.

.. code-block:: python

    m = nn.Module([X,y], [loss])

The two arguments to the `Module` constructor are a list of inputs and a list of outputs, respectively.

Now, we can split the data along the zeroth apply the module `m` separately to each piece.

.. code-block:: python

    split_loss = 0
    for start in xrange(0, batch_size, batch_size//4):
        sli = slice(start, start+batch_size//4)
        split_loss += m([X[sli], y[sli]])[0]
    split_loss /= 4
    params = nn.get_parameters(split_loss)
    gparams = cgt.grad(split_loss, params)
    updates2 = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
    updater =  cgt.function([X,y, stepsize], split_loss, updates=updates2)


Debugging
=========

Let's suppose you have compiled a ``function``, but it is failing at runtime or returning invalid results, and you're trying to figure out what might be wrong.
Here are some steps you can take:

1. Use the python backend. That is, modify ``.cgtrc`` or set ``CGT_FLAGS=backend=python`` at the command line. The python implementation of operations use numpy, which may catch certain errors (e.g. shape mismatches) that our C++ implementations miss.
2. Set the configuration variable ``debug=True`` (with ``.cgtrc`` or ``CGT_FLAGS``.) That will enable several pieces of functionality that store information in the graph when you are building it, so useful information can be printed when an exception is reached.
3. If there is a shape error, you can sometimes find it by using the ``infer_shape`` function. When constructing your variables, specify the known components of their shapes, e.g. Then you can add assertions with ``infer_shape`` (We plan to add more functionality soon that will catch shape errors at graph construction time.) Here's an example:
   
    .. code-block:: python

        x = cgt.matrix(fixed_shape=(None,5))
        y = cgt.matrix(fixed_shape=(5,4))
        z = x.dot(y)
        assert cgt.infer_shape(z)[1] == 4

You can also take this a step further and compute numerical values associated with the nodes in the graph. Just replace ``x`` and ``y`` above with constants, and then use ``cgt.simplify`` to compute the value of ``z``.

4. *Disable optimizations.* Set ``enable_simplification=False`` and ``enable_inplace_opt=False`` in ``.cgtrc``. Maybe there is a bug in cgt's optimization. If so, please :ref:`report it <BugsHelp>`.

Cookbook
========

See ``examples`` directory:

- ``demo_mnist.py``: shows how to build up a fully-connected or convolutional neural network  using a low-level API.
- ``demo_cifar.py``: train a convolutional neural net on CIFAR dataset using ``nn``'s Torch-like API.
- ``demo_char_rnn.py``: based on Andrej Karpathy's char-rnn code, but all in one file for building the deep LSTM or GRU model and generating text.
- ``demo_neural_turing_machine.py``: implementation of the `Neural Turing Machine <http://arxiv.org/abs/1410.5401>`_, with feedforward controller.

More examples are coming soon!

.. _BugsHelp:

Reporting Bugs and Getting Help
===============================

You can post your queries on the `cgt-users discussion group <https://groups.google.com/forum/#!forum/cgt-users>`_.
If you want to participate in the development of CGT, post on the `cgt-devel discussion group <https://groups.google.com/forum/#!forum/cgt-devel>`_.


.. _whynottheano:

Why not Build on Theano?
========================

CGT is heavily based on Theano, and we (the authors of CGT) think that Theano is a beautiful and highly innovative piece of software.
However, several limitation of Theano (in its current state) motivated us to consider creating a new library:

- **Problem**: Optimization and compilation of the graphs is very slow. For this reason, Theano becomes  inconvenient when working with large recurrent models. To use these models, one has to use the Scan operator, which is usually less convenient than constructing a graph with the unrolled computation. |br| **CGT solution**: (1) the main graph simplification process in CGT involves a single pass through the graph that applies several different types of replacement simultaneously (common subexpression elimination, constant propagation, arithmetic identities like ``x*1=x``.) In-place optimizations are performed in a second phase that also involves a single pass through the graph. Together, these phases take negligible time. Furthermore, we use a different graph data-structure (similar to SSA representations used by compilers) which allows for much cleaner simplification code. In Theano, the C++/CUDA compilation itself takes significant time, because Theano compiles a whole Python module (written in C++) for each function, which includes Python.h and numpy/arrayobject.h. On the other hand, CGT compiles a small C++ file with minimal header dependencies, taking a small fraction of a second, and the relevant function is later retrieved with ``dlopen`` and ``dlsym``.
- **Problem**: Theano can't straightforwardly be used to perform different operations in parallel, because of Python's GIL. |br| **CGT solution**: we create a representation of the computation called the execution graph, which can be executed in C++ independently of Python, and encodes all of the information necessary for concurrent execution of operations.
- **Problem**: When using GPUs, the user often obtains poor performance unless the user is careful to set up the graph in a way that the operations can be executed on the GPU. |br| **CGT solution**: we give the user finer grained control over which operation is performed on which device.
- **Problem**: Automatic upcasting rules (e.g. int * float = double) require the user to add casts many casting operations. |br| **CGT solution**: we globally choose either single or double (or quad) precision, using ``cgt.set_precision(...)``
- It is difficult to debug certain problems such as shape mismatches. Furthermore, Theano tensors have a `broadcastable` attribute that must be set to allow broadcasting and is point of confusion for many users. |br| **CGT solution**: we require explicit broadcasting using the ``broadcast(...)`` function. This requires slightly more verbosity but serves to eliminate many common errors and usually allows us to determine all of the shapes of intermediate variables in terms of the shapes of the inputs, which allows many shape errors to be caught at graph construction time.

Some of issues could be addressed within Theano's existing codebase, however, we believe that by breaking compatibility and starting from afresh, it will be possible to resolve them more cleanly.


.. |br| raw:: html

   <br />
