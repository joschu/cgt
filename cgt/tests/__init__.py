import cgt
from cgt.numeric_diff import numeric_grad, numeric_grad_multi
import numpy as np

def gradcheck_model(cost, params, extravars=(), extravals=(), atol=1e-8, eps=1e-9):
    precision = cgt.get_precision()
    if precision == "single":
        cgt.utils.warn("You're doing a gradient check with %s precision. Use double or better yet quad for best results"%(precision))
    assert all(isinstance(param, cgt.core.Input) for param in params)
    assert len(extravars) == len(extravals)

    # Convert to Argument nodes
    param_args = [cgt.core.Argument(typ=s.typ,name=s.name)if s.is_data() else s for s in params]

    # Get new cost in terms o farguments
    cost = cgt.core.clone(cost, replace=dict(zip(params,param_args)))

    grads = cgt.grad(cost, param_args)
    paramvals = [param.get_value() for param in params]
    fcost = cgt.function(param_args, cost, givens=zip(extravars,extravals))
    fgrad = cgt.function(param_args, grads,givens=zip(extravars,extravals))

    angrads = fgrad(*paramvals)
    nugrads = numeric_grad_multi(fcost, paramvals, eps=eps)

    for (angrad,nugrad) in zip(angrads,nugrads):
        assert np.allclose(angrad,nugrad,atol=atol)