import cgt
from cgt.numeric_diff import numeric_grad, numeric_grad_multi
from functools import wraps
import itertools
import numpy as np


def gradcheck_model(cost, params, extravars=(), extravals=(), atol=1e-8, eps=1e-9):
    precision = cgt.get_precision()
    if precision == "single":
        cgt.utils.warn("You're doing a gradient check with %s precision. Use double or better yet quad for best results"%(precision))
    assert all(param.is_input() for param in params)
    assert len(extravars) == len(extravals)

    # Convert to Argument nodes
    param_args = [cgt.core.Argument(typ=s.typ,name=s.name)if s.is_data() else s for s in params]

    # Get new cost in terms o farguments
    cost = cgt.core.clone(cost, replace=dict(zip(params,param_args)))

    grads = cgt.grad(cost, param_args)
    paramvals = [param.op.get_value() for param in params]
    fcost = cgt.function(param_args, cost, givens=zip(extravars,extravals))
    fgrad = cgt.function(param_args, grads,givens=zip(extravars,extravals))

    angrads = fgrad(*paramvals)
    nugrads = numeric_grad_multi(fcost, paramvals, eps=eps)

    for (angrad,nugrad) in zip(angrads,nugrads):
        assert np.allclose(angrad,nugrad,atol=atol)


def across_configs(*args, **kwargs):
    """
    Decorator for Nose test. Generates tests for all combinations of configuration options.

    Examples:

    Generates tests for all assignments of backends and precision.
      @across_configs
      def test_adagrad(): ...

    Generates tests for python/single and python/double:
      @across_configs(backends=("python",), precisions=("single", "double"))
      def test_adagrad(): ...
    """

    # If one function arg is passed, then apply this decorator with no parameters
    if len(args) == 1 and not kwargs and hasattr(args[0], "__call__"):
        return across_configs()(args[0])

    assert not args
    backends = kwargs.get("backends", ("python", "native"))
    precisions = kwargs.get("precisions", ("single", "double"))
    devtypes = kwargs.get("devtypes",("cpu",))
    pass_settings = kwargs.get("pass_settings", False)

    def decorator(check_func):
        @wraps(check_func)
        def check_func_with_config(backend, precision, devtype):
            with cgt.scoped_update_config(backend=backend, precision=precision, default_device=cgt.core.Device(devtype=devtype)):
                if pass_settings:
                    check_func(backend=backend, precision=precision)
                else:
                    check_func()

        @wraps(check_func_with_config)
        def wrapper():
            for backend, precision, devtype in itertools.product(backends, precisions, devtypes):
                yield check_func_with_config, backend, precision, devtype
        return wrapper

    return decorator
