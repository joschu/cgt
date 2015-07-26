def sgd_update(state, grad):
    state.theta[:] -= state.step_size * grad
