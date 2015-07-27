from __future__ import division
import cgt, nn
import numpy as np
from collections import namedtuple
import os.path as osp
import argparse
from time import time

# via https://github.com/karpathy/char-rnn/blob/master/model/GRU.lua
# via http://arxiv.org/pdf/1412.3555v1.pdf
def make_deep_gru(size_input, size_mem, n_layers, size_output):
    inputs = [cgt.matrix() for i_layer in xrange(n_layers+1)]
    outputs = []
    for i_layer in xrange(n_layers):
        prev_h = inputs[i_layer+1] # note that inputs[0] is the external input, so we add 1
        x = inputs[0] if i_layer==0 else outputs[i_layer-1]
        size_x = size_input if i_layer==0 else size_mem
        update_gate = cgt.sigmoid(
            nn.Affine(size_x, size_mem,name="i2u")(x)
            + nn.Affine(size_mem, size_mem, name="h2u")(prev_h))
        reset_gate = cgt.sigmoid(
            nn.Affine(size_x, size_mem,name="i2r")(x)
            + nn.Affine(size_mem, size_mem, name="h2r")(prev_h))
        gated_hidden = reset_gate * prev_h
        p2 = nn.Affine(size_mem, size_mem)(gated_hidden)
        p1 = nn.Affine(size_x, size_mem)(x)
        hidden_target = cgt.tanh(p1+p2)
        next_h = (1.0-update_gate)*prev_h + update_gate*hidden_target
        outputs.append(next_h)
    category_activations = nn.Affine(size_mem, size_output,name="pred")(outputs[-1])
    logprobs = nn.logsoftmax(category_activations)
    outputs.append(logprobs)

    return nn.Module(inputs, outputs)

def make_deep_lstm(size_input, size_mem, n_layers, size_output):
    inputs = [cgt.matrix() for i_layer in xrange(2*n_layers+1)]
    outputs = []
    for i_layer in xrange(n_layers):
        prev_h = inputs[i_layer*2]
        prev_c = inputs[i_layer*2+1]
        if i_layer==0:
            x = inputs[0]
            size_x = size_input
        else:
            x = outputs[(i_layer-1)*2]
            size_x = size_mem
        input_sums = nn.Affine(size_x, 4*size_mem)(x) + nn.Affine(size_x, 4*size_mem)(prev_h)
        sigmoid_chunk = cgt.sigmoid(input_sums[:,0:3*size_mem])
        in_gate = sigmoid_chunk[:,0:size_mem]
        forget_gate = sigmoid_chunk[:,size_mem:2*size_mem]
        out_gate = sigmoid_chunk[:,2*size_mem:3*size_mem]
        in_transform = cgt.tanh(input_sums[:,3*size_mem:4*size_mem])
        next_c = forget_gate*prev_c + in_gate * in_transform
        next_h = out_gate*cgt.tanh(next_c)
        outputs.append(next_c)
        outputs.append(next_h)

    category_activations = nn.Affine(size_mem, size_output)(outputs[-1])
    logprobs = nn.logsoftmax(category_activations)
    outputs.append(logprobs)

    return nn.Module(inputs, outputs)


def flatcat(xs):
    return cgt.concatenate([x.flatten() for x in xs])

# - why does initial state persist between batches?
# - GRU code and categories outputs
# - zero initialization ?!

def rmsprop_update(grad, state):
    state.sqgrad[:] *= state.decay_rate
    np.square(grad, out=state.scratch) # scratch=g^2
    state.sqgrad[:] += state.scratch
    np.sqrt(state.sqgrad, out=state.scratch) # scratch = scaling
    np.divide(grad, state.scratch, out=state.scratch) # scratch = grad/scaling
    np.multiply(state.scratch, state.step_size, out=state.scratch)
    state.theta[:] -= state.scratch

def make_loss_and_grad(arch, size_input, size_output, size_mem, size_batch, n_layers, n_unroll):
    # symbolic variables

    x_tnk = cgt.tensor3(dtype=cgt.floatX)
    labels_tnk = cgt.tensor3(dtype='i8')
    make_network = make_deep_lstm if arch=="lstm" else make_deep_gru
    network = make_network(size_input, size_mem, n_layers, size_output)
    init_hiddens = [cgt.matrix() for _ in xrange(get_num_hiddens(arch, n_layers))]
    # TODO fixed sizes

    prev_hiddens = init_hiddens
    loss = 0
    for t in xrange(n_unroll):
        outputs = network.expand([x_tnk[t]] + prev_hiddens)
        prediction_logprobs = outputs[-1]
        # loss = loss + nn.categorical_negloglik(prediction_probs, labels_tnk[t]).sum()
        loss = loss - (prediction_logprobs*labels_tnk[t]).sum()
        prev_hiddens = outputs[:-1]

    loss = loss / (n_unroll * size_batch)

    params = network.get_parameters()
    gradloss = cgt.grad(loss, params)

    flatgrad = flatcat(gradloss)

    f_loss_and_grad = cgt.make_function([x_tnk, labels_tnk] + init_hiddens, [loss, flatgrad] + prev_hiddens, fixed_sizes=True)
    f_loss = cgt.make_function([x_tnk, labels_tnk] + init_hiddens, loss)

    assert len(init_hiddens) == len(prev_hiddens)

    return network, f_loss, f_loss_and_grad


class Table(object):
    pass

def make_rmsprop_state(theta, step_size, decay_rate):
    out = Table()
    out.__dict__.update(theta=theta, sqgrad=np.zeros_like(theta)+1e-6, scratch=np.empty_like(theta), 
        step_size=step_size, decay_rate=decay_rate)
    return out
    
class Loader(object):
    def __init__(self, data_dir, size_batch, n_unroll, split_fractions):
        input_file = osp.join(data_dir,"input.txt")
        preproc_file = osp.join(data_dir, "preproc.npz")
        run_preproc = not osp.exists(preproc_file) or osp.getmtime(input_file) > osp.getmtime(preproc_file)
        if run_preproc:
            text_to_tensor(input_file, preproc_file)
        data_file = np.load(preproc_file)
        self.vocab_mapping = {char:ind for (ind,char) in enumerate(data_file["chars"])}
        data = data_file["inds"]
        data = data[:data.shape[0] - (data.shape[0] % size_batch)].reshape(size_batch, -1).T # inds_tn
        n_batches = (data.shape[0]-1) // n_unroll 
        data = data[:n_batches*n_unroll+1]  # now t-1 is divisble by batch size
        self.n_unroll = n_unroll
        self.data = data

        self.n_train_batches = int(n_batches*split_fractions[0])
        self.n_test_batches = int(n_batches*split_fractions[1])
        self.n_val_batches = n_batches - self.n_train_batches - self.n_test_batches

        print "%i train batches, %i test batches, %i val batches"%(self.n_train_batches, self.n_test_batches, self.n_val_batches)

    @property
    def size_vocab(self):
        return len(self.vocab_mapping)

    def train_batches_iter(self):
        for i in xrange(self.n_train_batches):
            start = i*self.n_unroll
            stop = (i+1)*self.n_unroll
            yield ind2onehot(self.data[start:stop], self.size_vocab), ind2onehot(self.data[start+1:stop+1], self.size_vocab) # XXX


def ind2onehot(inds, n_cls):
    out = np.zeros(inds.shape+(n_cls,),cgt.floatX)
    out.flat[np.arange(inds.size)*n_cls + inds.ravel()] = 1
    return out


def text_to_tensor(text_file, preproc_file):
    with open(text_file,"r") as fh:
        text = fh.read()
    char2ind = {}
    inds = []
    for char in text:
        ind = char2ind.get(char, -1)
        if ind == -1:
            ind = len(char2ind)
            char2ind[char] = ind
        inds.append(ind)
    np.savez(preproc_file, inds = inds, chars = sorted(char2ind, key = lambda char : char2ind[char]))


def get_num_hiddens(arch, n_layers):
        return {"lstm" : 2 * n_layers, "gru" : n_layers}[arch]

def main():

    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--size_mem", type=int,default=64)
    parser.add_argument("--size_batch", type=int,default=64)
    parser.add_argument("--n_layers",type=int,default=3)
    parser.add_argument("--n_unroll",type=int,default=64)
    parser.add_argument("--step_size",type=float,default=2e-3)
    parser.add_argument("--decay_rate",type=float,default=0.95)
    parser.add_argument("--n_epochs",type=int,default=5)
    parser.add_argument("--arch",choices=["lstm","gru"],default="lstm")
    parser.add_argument("--grad_check",action="store_true")

    opt = parser.parse_args()

    cgt.set_precision("quad" if opt.grad_check else "single")

    assert opt.n_unroll > 1

    loader = Loader(opt.data_dir,opt.size_batch, opt.n_unroll, (.8,.1,.1))

    network, f_loss, f_loss_and_grad = make_loss_and_grad(opt.arch, loader.size_vocab, loader.size_vocab, opt.size_mem, 
        opt.size_batch, opt.n_layers, opt.n_unroll)

    params = network.get_parameters()
    th = nn.setup_contiguous_storage(params)
    th[:] = np.random.uniform(-0.08, 0.08, th.shape)

    def initialize_hiddens():
        return [np.zeros((opt.size_batch, opt.size_mem), cgt.floatX) for _ in xrange(get_num_hiddens(opt.arch, opt.n_layers))]

    if opt.grad_check:
        x,y = loader.train_batches_iter().next()
        prev_hiddens = initialize_hiddens()
        def f(thnew):
            thold = th.copy()
            th[:] = thnew
            loss = f_loss(x,y, *prev_hiddens)
            th[:] = thold
            return loss
        from numeric_diff import numeric_grad
        g_num = numeric_grad(f, th,eps=1e-10)
        _, g_anal,_,_ = f_loss_and_grad(x,y,*prev_hiddens)
        assert np.allclose(g_num, g_anal, atol=1e-4)
        print "Gradient check succeeded!"
        return

    optim_state = make_rmsprop_state(theta=th, step_size = opt.step_size, 
        decay_rate = opt.decay_rate)

    for iepoch in xrange(opt.n_epochs):
        losses = []
        tstart = time()
        print "starting epoch",iepoch
        prev_hiddens = initialize_hiddens()
        for (x,y) in loader.train_batches_iter():
            out = f_loss_and_grad(x,y, *prev_hiddens)
            loss = out[0]
            grad = out[1]
            np.clip(grad,-5,5,out=grad)
            prev_hiddens = out[2:]
            rmsprop_update(grad, optim_state)
            losses.append(loss)
        print "%.3f s/batch. avg loss = %.3f"%((time()-tstart)/len(losses), np.mean(losses))
        optim_state.step_size *= .95

if __name__ == "__main__":
    main()
