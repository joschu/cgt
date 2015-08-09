#pragma once
#include "cgt_common.h"
#include <vector>
#include <thread>
#include <string>

namespace cgt {
using std::vector;

// note: no-args initializers are only here because they're required by cython

class ByRefFunCl {
public:
    cgtByRefFun fptr;
    void* data;
    ByRefFunCl(cgtByRefFun fptr, void* data) : fptr(fptr), data(data) {}
    ByRefFunCl() : fptr(NULL), data(NULL) {}
    void operator()(cgtObject ** reads, cgtObject * write) {
        (*fptr)(data, reads, write);
    }
};

struct ByValFunCl {
public:
    cgtByValFun fptr;
    void* data;
    ByValFunCl(cgtByValFun fptr, void* data) : fptr(fptr), data(data) {}
    ByValFunCl() : fptr(NULL), data(NULL) {}
    cgtObject * operator()(cgtObject ** args) {
        return (*fptr)(data, args);
    }
};

class MemLocation {
public:
    MemLocation() : index(0) {}
    MemLocation(size_t index) : index(index) {}
    size_t get_index() {return index;}
    size_t index;
};

class Interpreter;

class Instruction {
public:
    Instruction(const std::string& repr) : repr_(repr) { }
    virtual void fire(Interpreter*)=0;
    virtual ~Instruction() {};
    const vector<MemLocation>& get_readlocs() const {
        return readlocs;
    }
    const MemLocation& get_writeloc() const {
        return writeloc;
    }
    const std::string& repr() const { return repr_; }
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    std::string repr_;
};

class ExecutionGraph {
public:
    ExecutionGraph(const vector<Instruction*>& instrs, size_t n_args, size_t n_locs)
    : instrs_(instrs), n_args_(n_args), n_locs_(n_locs) {}
    ~ExecutionGraph();
    const vector<Instruction*>& instrs() const {return instrs_;}
    size_t n_args() const {return n_args_;}
    size_t n_locs() const {return n_locs_;}
    size_t n_instrs() const {return instrs_.size();}
private:
    vector<Instruction*> instrs_; // owns, will delete
    size_t n_args_;
    size_t n_locs_;
};

class Interpreter {
public:
    // called by external code
    virtual cgtTuple * run(cgtTuple *)=0;
    // called by instructions:
    virtual cgtObject * get(MemLocation)=0;
    virtual void set(MemLocation, cgtObject *)=0;
    virtual cgtObject * getarg(int)=0;
    virtual ~Interpreter() {}
};

// pass by value because of cython
Interpreter* create_interpreter(ExecutionGraph*, vector<MemLocation> output_locs, bool parallel);

class LoadArgument : public Instruction  {
public:
    LoadArgument(const std::string& repr, int ind, MemLocation writeloc) : Instruction(repr), ind(ind), writeloc(writeloc) {}
    void fire(Interpreter*);
private:
    int ind;
    MemLocation writeloc;
};


class Alloc : public Instruction {
public:
    Alloc(const std::string& repr, cgtDtype dtype, vector<MemLocation> readlocs, MemLocation writeloc)
    : Instruction(repr), dtype(dtype), readlocs(readlocs), writeloc(writeloc) {}
    void fire(Interpreter*);
private:
    cgtDtype dtype;
    vector<MemLocation> readlocs;
    MemLocation writeloc;
};

class BuildTup : public Instruction {
public:
    BuildTup(const std::string& repr, vector<MemLocation> readlocs, MemLocation writeloc)
    : Instruction(repr), readlocs(readlocs), writeloc(writeloc) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
};

class ReturnByRef : public Instruction  {
public:
    ReturnByRef(const std::string& repr, vector<MemLocation> readlocs, MemLocation writeloc, ByRefFunCl closure)
    : Instruction(repr), readlocs(readlocs), writeloc(writeloc), closure(closure) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    ByRefFunCl closure;
};

class ReturnByVal : public Instruction  {
public:
    ReturnByVal(const std::string& repr, vector<MemLocation> readlocs, MemLocation writeloc, ByValFunCl closure)
    : Instruction(repr), readlocs(readlocs), writeloc(writeloc), closure(closure) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    ByValFunCl closure;
};


}
