from google.protobuf import text_format
from caffe_pb2 import *
import os.path as osp
import cgt
import nn
import numpy as np
# infile = "/Users/joschu/Src/caffe/examples/mnist/lenet.prototxt"
infile = "/Users/joschu/Src/caffe/models/bvlc_googlenet/train_val.prototxt"

with open(osp.expanduser(infile),"r") as fh:
    text = fh.read()
net = NetParameter()
text_format.Merge(text, net)


name2node = {}

cgt.set_float_type('float32')

if net.input: #pylint: disable=E1101
    assert len(net.input) == 1 #pylint: disable=E1101
    name2node[net.input[0]] = cgt.Argument(cgt.Tensor(cgt.floatX,4),net.input[0], fixed_shape=net.input_dim) #pylint: disable=E1101


# XXX super inefficient
def shape_eval(x):
    return cgt.numeric_eval(cgt.simplify(x),{})

for layer in net.layer: #pylint: disable=E1101
    if layer.phase==TRAIN:
        print "loading layer %s type=%s in=%s out=%s"%(layer.name, layer.type, layer.bottom, layer.top)
        output = None
        inputs = [name2node[name] for name in layer.bottom]
        if layer.type == "Data":
            tp = layer.transform_param
            crop_size = tp.crop_size
            chans = len(tp.mean_value)
            dp = layer.data_param
            batch_size = dp.batch_size
            output = [cgt.Argument(cgt.Tensor(cgt.floatX,4),layer.name, fixed_shape=(batch_size,chans,crop_size,crop_size)),
                      cgt.Argument(cgt.Tensor(cgt.intX,2),layer.name, fixed_shape=(batch_size, 1))]
        elif layer.type == "Convolution":
            X = inputs[0]
            param = layer.convolution_param
            kh,kw = (param.kernel_size, param.kernel_size) if param.HasField("kernel_size")\
                else (param.kernel_h, param.kernel_w)
            nchanin = shape_eval(cgt.size(X,0)).item()
            Wshape = (param.num_output, nchanin, kh, kw)
            Wname = layer.param[0].name or layer.name+":W"
            Wval = np.empty(Wshape, dtype=cgt.floatX)
            W = name2node[Wname] = cgt.shared(Wval, name=Wname, fixed_shape=True)
            bshape = (1, param.num_output, 1, 1)
            bname = layer.param[1].name or layer.name+":b"
            bval = np.empty(bshape, dtype=cgt.floatX)
            b = name2node[bname] = cgt.shared(bval, name=bname, fixed_shape=True)
            sh,sw = (param.stride, param.stride) if param.HasField("stride")\
                else (param.stride_h, param.stride_w)
            output = [cgt.broadcast("+",nn.conv2d(X, W, subsample=(sh,sw)), b, "xxxx,1x11")]
        elif layer.type == "Pooling":
            param = layer.pooling_param
            X = inputs[0]
            pool_type = {param.MAX : "max", param.AVE : "mean"}[param.pool]
            height_in,width_in = shape_eval(cgt.shape(X)[2:4])
            kernel = (param.kernel_size, param.kernel_size) if param.HasField("kernel_size")\
                else (param.kernel_h, param.kernel_w)
            stride = (param.stride, param.stride) if param.HasField("stride")\
                else (param.stride_h, param.stride_w)
            pad = (param.pad, param.pad) if param.HasField("pad")\
                else (param.pad_h, param.pad_w)
            output = [nn.pool(pool_type, X, stride, kernel, pad)]
        elif layer.type == "InnerProduct":
            X = inputs[0]
            if X.ndim == 4:
                X = cgt.reshape(X, [X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]] )
            param = layer.inner_product_param
            nchanin = shape_eval(X.shape[1])
            Wshape = (param.num_output, nchanin)
            Wname = layer.param[0].name or layer.name+":W"
            Wval = np.empty(Wshape, dtype=cgt.floatX)
            W = name2node[Wname] = cgt.shared(Wval, name=Wname, fixed_shape=True)
            bshape = (1, param.num_output)
            bname = layer.param[1].name or layer.name+":b"
            bval = np.empty(bshape, dtype=cgt.floatX)
            b = name2node[bname] = cgt.shared(bval, name=bname, fixed_shape=True)
            yname = layer.top[0]
            output = [cgt.broadcast("+",X.dot(W), b, "xx,1x")          ]
        elif layer.type == "ReLU":
            output = [nn.relu(inputs[0])]
        elif layer.type == "Softmax":
            output = [nn.softmax(inputs[0])]
        elif layer.type == "LRN":
            # XXX needs params
            param = layer.lrn_param
            output = [nn.lrn(inputs[0], param.alpha,param.beta, param.local_size)]
        elif layer.type == "Concat":
            param = layer.concat_param
            output = [cgt.concatenate(inputs, param.concat_dim)            ]
        elif layer.type == "Dropout":
            output = [nn.dropout(inputs[0])]
        elif layer.type == "SoftmaxWithLoss":
            output = [nn.loglik_softmax(inputs[0], inputs[1])]
        elif layer.type == "Accuracy":
            output = [nn.zero_one_loss(inputs[0], inputs[1])]
        else:
            cgt.error("unrecognized layer type %s"%layer.type)

        assert output is not None

        # assert isinstance(output, cgt.Node)
        for i in xrange(len(layer.top)): name2node[layer.top[i]] = output[i]
        print "stored", layer.top[0]
        if layer.type != "Data":
            print "shape",layer.type, shape_eval(cgt.shape(name2node[layer.bottom[0]])), shape_eval(cgt.shape(name2node[layer.top[0]]))




