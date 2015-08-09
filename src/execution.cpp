#include "execution.h"
#include <cstdio>
#include <queue>
#include <set>
#include <vector>

namespace cgt {

template <typename T>
T idx(cgtArray * x, size_t a) {
    return ((T*)x->data())[a];
}

ExecutionGraph::~ExecutionGraph() {
    for (auto instr : instrs_) delete instr;
}

class SequentialInterpreter : public Interpreter {
public:
    SequentialInterpreter(ExecutionGraph* eg, const vector<MemLocation>& output_locs) 
    : eg_(eg), output_locs_(output_locs), storage_(eg->n_locs()), args_(NULL) { }

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
            // printf("Firing %s\n", instr->repr().c_str());
            instr->fire(this);
        }
        args_ = NULL;
        size_t n_outputs = output_locs_.size();
        cgtTuple * out = new cgtTuple(n_outputs);
        for (int i=0; i < n_outputs; ++i) {
            int index = output_locs_[i].index; // XXX what is this used for?
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

class ParallelInterpreter: public Interpreter {
public:
    ParallelInterpreter(ExecutionGraph* eg, const vector<MemLocation>& output_locs)
    : eg_(eg), output_locs_(output_locs), storage_(eg->n_locs()), args_(NULL), write_queue_(eg->n_locs()) {

    }

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

        setup_instr_locs();
        while (instrs_left_.size() > 0) {
            const int nthreads = ready_instr_inds_.size();
            // XXX use thread pool
            std::vector<std::thread> instr_threads(nthreads);
            for (int instr_ind : ready_instr_inds_) {
                Instruction* instr = eg_->instrs()[instr_ind];
                instr_threads[instr_ind] = std::thread(std::bind(&Instruction::fire, instr,
                            std::placeholders::_1), this);
            }
            for (int k = 0; k < instr_threads.size(); k++)
                instr_threads[k].join();
            update_instr_locs();
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
    void setup_instr_locs() {
        for (int k=0; k < eg_->n_instrs(); k++) {
            write_queue_[k].push(eg_->instrs()[k]->get_writeloc().index);
        }
        update_instr_locs();
    }
    void update_instr_locs() {
        ready_instr_inds_.clear();
        for (int instr_ind : instrs_left_) {
            if (ready_to_fire(instr_ind))
                ready_instr_inds_.push_back(instr_ind);
        }
    }
    bool ready_to_fire(int instr_ind) {
        Instruction* instr = eg_->instrs()[instr_ind];
        bool read_rdy = true;
        for (MemLocation loc : instr->get_readlocs())
            if (write_queue_[loc.index].size() > 0)
                read_rdy = false;
        bool write_rdy = write_queue_[instr->get_writeloc().index].front() == instr_ind;
        return read_rdy && write_rdy;
    }
    ~ParallelInterpreter() {
    }
private:
    ExecutionGraph* eg_;
    vector<MemLocation> output_locs_;
    vector<IRC<cgtObject>> storage_;
    std::set<int> instrs_left_;
    vector<int> ready_instr_inds_;
    vector<std::queue<int> > write_queue_;
    cgtTuple * args_;
};

Interpreter* create_interpreter(ExecutionGraph* eg, vector<MemLocation> output_locs, bool parallel) {
    if (parallel) {
        return new ParallelInterpreter(eg, output_locs);
    }
    else{
        return new SequentialInterpreter(eg, output_locs);
    }
}

void LoadArgument::fire(Interpreter* interp) {
    interp->set(writeloc, interp->getarg(ind));
}

void Alloc::fire(Interpreter* interp) {
    int ndim = readlocs.size();
    std::vector<size_t> shape(ndim);
    for (int i=0; i < ndim; ++i) {
        cgtArray * sizeval = (cgtArray *)interp->get(readlocs[i]);
        cgt_assert(sizeval->dtype() == cgt_i8);
        shape[i] = idx<size_t>(sizeval, 0); 
    }

    cgtArray* cur = static_cast<cgtArray*>(interp->get(writeloc));
    if (!(cur && cur->ndim() == ndim && std::equal(shape.begin(), shape.end(), cur->shape()))) {
        interp->set(writeloc, new cgtArray(ndim, shape.data(), dtype, cgtCPU));
    }
}

void BuildTup::fire(Interpreter* interp) {
    cgtTuple * out = new cgtTuple(readlocs.size());
    for (int i=0; i < readlocs.size(); ++i) {
        out->setitem(i, interp->get(readlocs[i]));
    }
    interp->set(writeloc, out);
}

void ReturnByRef::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    cgtObject * reads[n_inputs];
    for (int i=0; i < n_inputs; ++i) {
        reads[i] = interp->get(readlocs[i]);
    }
    cgtObject * write = interp->get(writeloc);
    closure(reads, write);
}

// TODO actually allocate tuple
void ReturnByVal::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    vector<cgtObject *> args(n_inputs);
    for (int i = 0; i < n_inputs; ++i) {
        args[i] = interp->get(readlocs[i]);
    }    
    interp->set(writeloc, closure(args.data())); // XXX
}

}
