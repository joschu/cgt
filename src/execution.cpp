#include "execution.h"
#include <cstdio>

namespace cgt {

template <typename T>
T idx(cgtArray * x, size_t a) {
    return ((T*)x->data)[a];
}

ExecutionGraph::~ExecutionGraph() {
    for (auto instr : instrs_) delete instr;
}

class SequentialInterpreter : public Interpreter {
public:
    SequentialInterpreter(ExecutionGraph* eg, const vector<MemLocation>& output_locs) 
    : eg_(eg), output_locs_(output_locs), storage_(eg->n_locs()), args_(NULL) {}

    cgtObject * get(MemLocation m) {
        return storage_[m.index].get();
    }
    void set(MemLocation m, cgtObject * val) {
        storage_[m.index] = val;
    }
    cgtObject * getarg(int argind) {
        cgt_assert(argind < args_->len);        
        return args_->members[argind].get();
    }
    cgtTuple * run(cgtTuple * newargs) {
        args_ = newargs;
        cgt_assert(newargs != NULL);
        cgt_assert(newargs->len == eg_->n_args());
        for (Instruction* instr : eg_->instrs()) {
            instr->fire(this);
        }
        args_ = NULL;
        size_t n_outputs = output_locs_.size();
        cgtTuple * out = new cgtTuple(n_outputs);
        for (int i=0; i < n_outputs; ++i) {
            int index = output_locs_[i].index;
            out->setitem(i, get(output_locs_[i]));
        }
        return out;
        // todo actually do something with outputs
    }
    ~SequentialInterpreter() {
    }
private:
    ExecutionGraph* eg_;
    vector<MemLocation> output_locs_;
    vector<IRC<cgtObject>> storage_;
    cgtTuple * args_;
};

Interpreter* create_interpreter(ExecutionGraph* eg, vector<MemLocation> output_locs) {
    return new SequentialInterpreter(eg, output_locs);
}

void LoadArgument::fire(Interpreter* interp) {
    interp->set(writeloc, interp->getarg(ind));
}

bool array_equal(int n, size_t* x, size_t* y) {
    bool out = true;
    for (int i=0; i < n; ++i) out = out && (x[i]==y[i]);
    return out;
}

void Alloc::fire(Interpreter* interp) {
    int ndim = readlocs.size();
    size_t shape[ndim];
    for (int i=0; i < readlocs.size(); ++i) {
        cgtArray * sizeval = (cgtArray *)interp->get(readlocs[i]);
        cgt_assert(sizeval->dtype == cgt_i8);
        shape[i] = idx<size_t>(sizeval, 0); 
    }

    cgtArray* cur = static_cast<cgtArray*>(interp->get(writeloc));
    if (!(cur && array_equal(ndim, cur->shape, shape))) {
        interp->set(writeloc, new cgtArray(ndim, shape, dtype, cgtCPU));        
    }

}

void BuildTup::fire(Interpreter* interp) {
    cgtTuple * out = new cgtTuple(readlocs.size());
    for (int i=0; i < readlocs.size(); ++i) {
        out->setitem(i, interp->get(readlocs[i]));
    }
    interp->set(writeloc, out);
}

void CallByRef::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    cgtObject * reads[n_inputs];
    for (int i=0; i < n_inputs; ++i) {
        reads[i] = interp->get(readlocs[i]);
    }
    cgtObject * write = interp->get(writeloc);
    closure(reads, write);
}

// TODO actually allocate tuple
void CallByVal::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    vector<cgtObject *> args(n_inputs);
    for (int i = 0; i < n_inputs; ++i) {
        args[i] = interp->get(readlocs[i]);
    }    
    interp->set(writeloc, closure(args.data())); // XXX
}

}