#include "execution.h"
#include <cstdio>

namespace cgt {

template <typename T>
T idx(Array * x, size_t a) {
    return ((T*)x->data)[a];
}

ExecutionGraph::~ExecutionGraph() {}

class SequentialInterpreter : public Interpreter {
public:
    SequentialInterpreter(ExecutionGraph* eg) : 
        eg(eg),
        storage(eg->n_mem_locs()),
        args(NULL)
    {
    }

    Object * get(MemLocation m) {
        return storage[m.index].get();
    }
    void set(MemLocation m, Object * val) {
        storage[m.index] = val;
    }
    Object * getarg(int argind) {
        cgt_assert(argind < args->len);        
        return args->members[argind].get();
    }
    Tuple * run(Tuple * newargs) {
        cgt_assert(newargs != NULL);
        cgt_assert(newargs->len == eg->n_args());
        args = newargs;
        for (Instruction* instr : eg->get_instrs()) {
            instr->fire(this);
        }
        args = NULL;
        Tuple * out = new Tuple(eg->n_outputs());
        for (int i=0; i < eg->n_outputs(); ++i) {
            int index = eg->get_output_locs()[i].index;
            out->setitem(i, get(eg->get_output_locs()[i]));
        }
        return out;
        // todo actually do something with outputs
    }
    ~SequentialInterpreter() {
    }
private:
    ExecutionGraph* eg;
    vector<IRC<Object>> storage;
    Tuple * args;
};

Interpreter* create_interpreter(ExecutionGraph* eg) {
    return new SequentialInterpreter(eg);
}

void LoadArgument::fire(Interpreter* interp) {
    interp->set(writeloc, interp->getarg(ind));
}

void Alloc::fire(Interpreter* interp) {
    int ndim = readlocs.size();
    size_t shape[ndim];
    for (int i=0; i < readlocs.size(); ++i) {
        Array * sizeval = (Array *)interp->get(readlocs[i]);
        cgt_assert(sizeval->dtype == cgt_i8);
        shape[i] = idx<size_t>(sizeval, 0); 
    }
    interp->set(writeloc, new Array(ndim, shape, dtype, DevCPU));
}

void BuildTup::fire(Interpreter* interp) {
    Tuple* out = new Tuple(readlocs.size());
    for (int i=0; i < readlocs.size(); ++i) {
        out->setitem(i, interp->get(readlocs[i]));
    }
    interp->set(writeloc, out);
}

void InPlace::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    Object * reads[n_inputs];
    for (int i=0; i < n_inputs; ++i) {
        reads[i] = interp->get(readlocs[i]);
    }
    Object* write = interp->get(writeloc);
    closure(reads, write);
}

// TODO actually allocate tuple
void ValReturning::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    vector<Object *> args(n_inputs);
    for (int i = 0; i < n_inputs; ++i) {
        args[i] = interp->get(readlocs[i]);
    }    
    interp->set(writeloc, closure(args.data())); // XXX
}

}