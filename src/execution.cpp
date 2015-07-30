#include "execution.h"
#include <cstdio>

namespace cgt {

template <typename T>
T idx(cgt_array* x, size_t a) {
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

    cgt_object* get(MemLocation m) {
        return storage[m.index].get();
    }
    void set(MemLocation m, cgt_object* val) {
        storage[m.index] = val;
    }
    cgt_object* getarg(int argind) {
        cgt_assert(argind < args->len);        
        return args->members[argind].get();
    }
    cgt_tuple* run(cgt_tuple* newargs) {
        cgt_assert(newargs != NULL);
        cgt_assert(newargs->len == eg->n_args());
        args = newargs;
        for (Instruction* instr : eg->get_instrs()) {
            instr->fire(this);
        }
        args = NULL;
        cgt_tuple* out = new cgt_tuple(eg->n_outputs());
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
    vector<IRC<cgt_object>> storage;
    cgt_tuple* args;
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
        cgt_array* sizeval = (cgt_array*)interp->get(readlocs[i]);
        cgt_assert(sizeval->dtype == cgt_i8);
        shape[i] = idx<size_t>(sizeval, 0); 
    }
    interp->set(writeloc, new cgt_array(ndim, shape, dtype, cgt_cpu));
}

void InPlace::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    cgt_object* args[n_inputs+1];
    for (int i=0; i < n_inputs; ++i) {
        args[i] = interp->get(readlocs[i]);
    }
    args[n_inputs] = interp->get(writeloc);
    closure(args);
}

// TODO actually allocate tuple
void ValReturning::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    vector<cgt_object*> args(n_inputs);
    for (int i = 0; i < n_inputs; ++i) {
        args[i] = interp->get(readlocs[i]);
    }    
    interp->set(writeloc, closure(args.data())); // XXX
}

}