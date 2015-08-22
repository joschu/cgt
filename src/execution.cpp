#include "execution.h"
#include <cstdio>
#include <queue>
#include <set>
#include <vector>
#include "util/ThreadPool.h"
#include "unistd.h"

namespace cgt {

using std::vector;
using std::atomic;

// ================================================================
// Sequential interpreter
// ================================================================

class SequentialInterpreter : public Interpreter {
public:
    SequentialInterpreter(ExecutionGraph* eg, const vector<MemLocation>& output_locs) 
    : eg_(eg), output_locs_(output_locs), storage_(eg->n_locs()), args_(NULL) { }

    cgtObject * get(const MemLocation& m) {
        return storage_[m.index()].get();
    }
    void set(const MemLocation& m, cgtObject * val) {
        storage_[m.index()] = val;
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
            int index = output_locs_[i].index(); // XXX what is this used for?
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


// ================================================================
// Parallel interpreter
// ================================================================

/**


Pseudocode for run():

    Launch the instructions with no prerequisites
    while (npending > 0) sleep

    when instruction finishes, increment its write counter
    launch all instructions that are ready and read from write loc

*/

class ParallelInterpreter: public Interpreter {
public:
    using InstrInd = size_t;    
    ParallelInterpreter(ExecutionGraph* eg, const vector<MemLocation>& output_locs);

    cgtObject * get(const MemLocation& m) {
        return storage_[m.index()].get();
    }
    void set(const MemLocation& m, cgtObject * val) {
        storage_[m.index()] = val;
    }
    cgtObject * getarg(int argind) {
        cgt_assert(argind < args_->len);
        return args_->members[argind].get();
    }

    cgtTuple * run(cgtTuple * newargs);
    bool instr_is_ready(InstrInd ind);
    void fire_instr(InstrInd instr_ind);
    void trigger_instr(InstrInd instr_ind);

    ~ParallelInterpreter() {}
private:
    ExecutionGraph* eg_;
    vector<MemLocation> output_locs_;
    vector<IRC<cgtObject>> storage_;
    cgtTuple * args_;
    ThreadPool pool_;
    vector<int> n_writes_total_; // home many writes total to each mem location
    vector<InstrInd> no_prereqs_; // instructions with no prereqs
    vector<vector<InstrInd>> loc2rwinstrs_; // mem loc idx to readers
    vector<int> instr2writecount_; // instr idx 2 what write count should be when its fired

    std::mutex dont_fire_twice_;

    atomic<int> n_pending_; // number of pending instrs
    atomic<int> n_total_;
    vector<atomic<bool>> instr_triggered_; // bool whether instr was triggered or not
    vector<atomic<int>> n_writes_sofar_; // how many writes so far to each mem location
};

ParallelInterpreter::ParallelInterpreter(ExecutionGraph* eg, const vector<MemLocation>& output_locs)
  : eg_(eg), 
    output_locs_(output_locs), 
    storage_(eg->n_locs()), 
    args_(NULL),
    pool_(int(std::thread::hardware_concurrency())),
    n_writes_total_(eg->n_locs()), 
    no_prereqs_(),
    loc2rwinstrs_(eg->n_locs()),
    instr2writecount_(eg->n_instrs()),
    n_pending_(0),
    n_total_(0),
    instr_triggered_(eg->n_instrs()),
    n_writes_sofar_(eg->n_locs()) 
{
    InstrInd instr_ind=0;
    for (auto& instr : eg_->instrs()) {
        auto write_ind = instr->get_writeloc().index();
        if (instr->get_readlocs().empty()) {
            no_prereqs_.push_back(instr_ind); // xxx this is wrong if the write loc needs to be allocated. e.g. if Constant was byref
        }
        else {
            for (auto& m : instr->get_readlocs()) {
                loc2rwinstrs_[m.index()].push_back(instr_ind);
            }
        }
        instr2writecount_[instr_ind] = n_writes_total_[write_ind];
        loc2rwinstrs_[write_ind].push_back(instr_ind);
        n_writes_total_[write_ind] += 1;
        ++instr_ind;
    }
}


bool ParallelInterpreter::instr_is_ready(InstrInd instr_ind) {
    bool result=true;
    auto instr = eg_->instrs()[instr_ind];
    for (auto readloc : instr->get_readlocs()) {
        result = result && (n_writes_sofar_[readloc.index()] == n_writes_total_[readloc.index()]);
    }
    result = result && (n_writes_sofar_[instr->get_writeloc().index()] == instr2writecount_[instr_ind]);
    // cgt_assert(get(instr->get_writeloc())!= NULL);
    return result;
}

cgtTuple * ParallelInterpreter::run(cgtTuple * newargs) {
    args_ = newargs;
    cgt_assert(newargs != NULL);
    cgt_assert(newargs->len == eg_->n_args());

    // setup
    n_pending_=0;
    n_total_=0;
    for (auto& n : n_writes_sofar_) { n = 0;}
    for (int i=0; i < eg_->n_instrs(); ++i) instr_triggered_[i] = false;

    for (InstrInd i : no_prereqs_) {
        trigger_instr(i);
        auto instr = eg_->instrs()[i];
        cgtObject* objptr = get(instr->get_writeloc());
        // printf("init: instr %d triggered: %s %zu\n", i, instr->repr().c_str(), objptr);

    }

    while (n_pending_ > 0) {
        usleep(1000);
    }

    args_ = NULL;
    size_t n_outputs = output_locs_.size();
    cgtTuple * out = new cgtTuple(n_outputs);
    for (int i=0; i < n_outputs; ++i) {
        int index = output_locs_[i].index();
        out->setitem(i, get(output_locs_[i]));
    }

    printf("don %i insrts \n", (int)n_total_);


    return out;
}

void ParallelInterpreter::fire_instr(InstrInd instr_ind)
{
    printf("runing instr %i\n", instr_ind);
    Instruction* instr = eg_->instrs()[instr_ind];
    instr->fire(this);
    // printf("instr %d DONE: %s\n", instr_ind, instr->repr().c_str());

    InstrInd write_ind = instr->get_writeloc().index();
    n_writes_sofar_[write_ind] += 1;
    for (auto& nextinstr_ind : loc2rwinstrs_[write_ind]) {
        printf("%i finished. checking %i\n", instr_ind, nextinstr_ind);
        if (instr_is_ready(nextinstr_ind)) {            
            { // can we cleverly do this without a lock? we just need a loc specific to instr
                // e.g. just write tid into atomic
                std::lock_guard<std::mutex> lock(dont_fire_twice_);
                if (!instr_triggered_[nextinstr_ind]) {
                    // Instruction* nextinstr = eg_->instrs()[nextinstr_ind];
                    // cgtObject* objptr = get(nextinstr->get_writeloc());
                    // printf("instr %d triggered: %s %zu\n", nextinstr_ind, nextinstr->repr().c_str(), objptr);
                    trigger_instr(nextinstr_ind);                    

                }
            }
        }
    } 

    --n_pending_;
}

void ParallelInterpreter::trigger_instr(InstrInd instr_ind)
{
    // printf("instr %d ENQUED: %s\n", instr_ind, eg_->instrs()[instr_ind]->repr().c_str());    
    ++n_pending_;
    ++n_total_;
    pool_.enqueue(&ParallelInterpreter::fire_instr, this, instr_ind);
    instr_triggered_[instr_ind] = true;                            
}


// ================================================================
// Instructions
// ================================================================

void LoadArgument::fire(Interpreter* interp) {
    interp->set(writeloc, interp->getarg(ind));
}

void Alloc::fire(Interpreter* interp) {
    int ndim = readlocs.size();
    vector<size_t> shape(ndim);
    for (int i=0; i < ndim; ++i) {
        cgtArray * sizeval = (cgtArray *)interp->get(readlocs[i]);
        cgt_assert(sizeval->dtype() == cgt_i8);
        shape[i] = sizeval->at<size_t>(0);
    }

    cgtArray* cur = static_cast<cgtArray*>(interp->get(writeloc));
    if (!(cur && cur->ndim() == ndim && std::equal(shape.begin(), shape.end(), cur->shape()))) {
        interp->set(writeloc, new cgtArray(ndim, shape.data(), dtype, writeloc.devtype()));
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
    callable(reads, write);
}

// TODO actually allocate tuple
void ReturnByVal::fire(Interpreter* interp) {
    int n_inputs = readlocs.size();
    vector<cgtObject *> args(n_inputs);
    for (int i = 0; i < n_inputs; ++i) {
        args[i] = interp->get(readlocs[i]);
    }    
    interp->set(writeloc, callable(args.data())); // XXX
}


// ================================================================
// Misc
// ================================================================


ExecutionGraph::~ExecutionGraph() {
    for (auto instr : instrs_) delete instr;
}

Interpreter* create_interpreter(ExecutionGraph* eg, vector<MemLocation> output_locs, bool parallel) {
    if (parallel) {
        return new ParallelInterpreter(eg, output_locs);
    }
    else{
        return new SequentialInterpreter(eg, output_locs);
    }
}


}