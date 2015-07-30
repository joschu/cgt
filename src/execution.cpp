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
        args(NULL),
        output(new_cgt_tuple(eg->n_outputs()))
    {
    }

    cgt_object* get(MemLocation m) {
        return storage[m.index];
    }
    void set(MemLocation m, cgt_object* val) {
        storage[m.index] = val;
    }
    cgt_object* getarg(int argind) {
        cgt_assert(argind < args->len);        
        return args->members[argind];
    }
    cgt_tuple* run(cgt_tuple* newargs) {
        cgt_assert(newargs != NULL);
        cgt_assert(newargs->len == eg->n_args());
        args = newargs;
        for (Instruction* instr : eg->get_instrs()) {
            instr->fire(this);
        }
        args = NULL;
        for (int i=0; i < eg->n_outputs(); ++i) {
            int index = eg->get_output_locs()[i].index;
            output->members[i] = get(eg->get_output_locs()[i]);
        }
        return output;
        // todo actually do something with outputs
    }
    void alloc_array(MemLocation mem, int ndim, size_t* shape, cgt_dtype dtype) override {
        set(mem, (cgt_object*)new_cgt_array(ndim, shape, dtype, cgt_cpu)); // XXX
    }
    void alloc_tuple(MemLocation mem, int len) override {
        set(mem, (cgt_object*)new_cgt_tuple(len)); // XXX
    }
    ~SequentialInterpreter() {
        delete output;
    }
private:
    ExecutionGraph* eg;
    vector<cgt_object*> storage;
    cgt_tuple* args;
    cgt_tuple* output;
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
    // interp->set(writeloc, cgt_alloc_array(ndim, shape, this->dtype));    
    interp->alloc_array(writeloc, ndim, shape, this->dtype);
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
    vector<cgt_object*> args(n_inputs+1);
    for (int i = 0; i < n_inputs; ++i) {
        args[i] = interp->get(readlocs[i]);
    }
    args[n_inputs] = interp->get(writeloc);
    closure(args.data());
    interp->set(writeloc, args[n_inputs]); // XXX
}

}