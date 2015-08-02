#pragma once
#include "cgt_common.h"
#include <vector>

namespace cgt {
using std::vector;

// note: no-args initializers are only here because they're required by cython

class InPlaceFunCl {
public:
    Inplacefun fptr;
    void* data;
    InPlaceFunCl(Inplacefun fptr, void* data) : fptr(fptr), data(data) {}
    InPlaceFunCl() : fptr(NULL), data(NULL) {}
    void operator()(Object** reads, Object* write) {
        (*fptr)(data, reads, write);
    }
};

struct ValRetFunCl {
public:
    Valretfun fptr;
    void* data;
    ValRetFunCl(Valretfun fptr, void* data) : fptr(fptr), data(data) {}
    ValRetFunCl() : fptr(NULL), data(NULL) {}
    Object * operator()(Object** args) {
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
    virtual void fire(Interpreter*)=0;
    virtual ~Instruction() {};
};

class ExecutionGraph {
public:
    ExecutionGraph(const vector<Instruction*>& instrs, size_t n_args, size_t n_locs)
    : instrs_(instrs), n_args_(n_args), n_locs_(n_locs) {}
    ~ExecutionGraph();
    const vector<Instruction*>& instrs() const {return instrs_;}
    size_t n_args() const {return n_args_;}
    size_t n_locs() const {return n_locs_;}
private:
    vector<Instruction*> instrs_; // owns, will delete
    size_t n_args_;
    size_t n_locs_;
};

class Interpreter {
public:
    // called by external code
    virtual Tuple * run(Tuple *)=0;
    // called by instructions:
    virtual Object * get(MemLocation)=0;
    virtual void set(MemLocation, Object *)=0;
    virtual Object * getarg(int)=0;
    virtual ~Interpreter() {}
};

// pass by value because of cython
Interpreter* create_interpreter(ExecutionGraph*, vector<MemLocation> output_locs);

class LoadArgument : public Instruction  {
public:
    LoadArgument(int ind, MemLocation writeloc) : ind(ind), writeloc(writeloc) {}
    void fire(Interpreter*);
private:
    int ind;
    MemLocation writeloc;
};


class Alloc : public Instruction {
public:
    Alloc(Dtype dtype, vector<MemLocation> readlocs, MemLocation writeloc)
    : dtype(dtype), readlocs(readlocs), writeloc(writeloc) {}
    void fire(Interpreter*);
private:
    Dtype dtype;
    vector<MemLocation> readlocs;
    MemLocation writeloc;
};

class BuildTup : public Instruction {
public:
    BuildTup(vector<MemLocation> readlocs, MemLocation writeloc)
    : readlocs(readlocs), writeloc(writeloc) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
};

class InPlace : public Instruction  {
public:
    InPlace(vector<MemLocation> readlocs, MemLocation writeloc, InPlaceFunCl closure)
    : readlocs(readlocs), writeloc(writeloc), closure(closure) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    InPlaceFunCl closure;
};

class ValReturning : public Instruction  {
public:
    ValReturning(vector<MemLocation> readlocs, MemLocation writeloc, ValRetFunCl closure)
    : readlocs(readlocs), writeloc(writeloc), closure(closure) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    ValRetFunCl closure;
};


}
