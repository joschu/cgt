#include "execution.h"
#include <shared_ptr>

class SequentialInterpreter : public Interpreter {
public:
    SequentialInterpreter(std::shared_ptr<ExecutionGraph> eg) : 
        m_eg(eg),
        m_storage(eg.n_locs),
        m_args(NULL) {}

    cgt_object* get(MemLocation m) {
        return m_storage[m.index]
    }
    void set(MemLocation m, cgt_object* val) {
        m_storage[m.index] = val;
    }
    void getarg(int argind) {
        cgt_assert(argind < m_args->len);
        return m_args->members[argind];
    }
    void operator()(cgt_tuple*, cgt_tuple*) {
        for (Instruction* instr : m_eg->get_instrs()) {
            instr->fire(this);
        }
        TODO_WHATS_THE_OUTPUT
    }
private:
    shared_ptr<ExecutionGraph> m_eg;
    vector<cgt_object*> m_storage;
    cgt_tuple* m_args;
};