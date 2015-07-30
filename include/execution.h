#pragma once
#include "cgt_common.h"
#include <vector>

namespace cgt {
using std::vector;

// note: no-args initializers are only here because they're required by cython

class InPlaceFun {
public:
    cgt_inplacefun fptr;
    void* data;
    InPlaceFun(cgt_inplacefun fptr, void* data) : fptr(fptr), data(data) {}
    InPlaceFun() : fptr(NULL), data(NULL) {}
    void operator()(cgt_object** args) {
        (*fptr)(data, args);
    }
};

struct ValRetFun {
public:
    cgt_valretfun fptr;
    void* data;
    ValRetFun(cgt_valretfun fptr, void* data) : fptr(fptr), data(data) {printf("value %zui\n", ((size_t*)data)[0]);}
    ValRetFun() : fptr(NULL), data(NULL) {}
    cgt_object* operator()(cgt_object** args) {
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
};

class ExecutionGraph {
public:
    ExecutionGraph(int n_args, int n_locs, vector<MemLocation> output_locs)
    : n_args_(n_args), locs(n_locs), output_locs(output_locs) {
        for (int i=0; i < n_locs; ++i) locs[i].index = i;
    }
    int n_outputs() {return output_locs.size();}    
    int n_args() {return n_args_;}
    int n_mem_locs() {return locs.size();}
    void add_instr(Instruction* instr) {instrs.push_back(instr);}    
    const vector<Instruction*>& get_instrs() {return instrs;}
    const vector<MemLocation>& get_output_locs() {return output_locs;}
    ~ExecutionGraph();
private:
    int n_args_;
    vector<MemLocation> locs;
    vector<MemLocation> output_locs;
    vector<Instruction*> instrs;
};

class Interpreter {
public:
    // called by external code
    virtual cgt_tuple* run(cgt_tuple*)=0;
    // called by instructions:
    virtual cgt_object* get(MemLocation)=0;
    virtual void set(MemLocation, cgt_object*)=0;
    virtual cgt_object* getarg(int)=0;
};

Interpreter* create_interpreter(ExecutionGraph*);


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
    Alloc(cgt_dtype dtype, vector<MemLocation> readlocs, MemLocation writeloc)
    : dtype(dtype), readlocs(readlocs), writeloc(writeloc) {}
    void fire(Interpreter*);
private:
    cgt_dtype dtype;
    vector<MemLocation> readlocs;
    MemLocation writeloc;
};

class InPlace : public Instruction  {
public:
    InPlace(vector<MemLocation> readlocs, MemLocation writeloc, InPlaceFun closure)
    : readlocs(readlocs), writeloc(writeloc), closure(closure) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    InPlaceFun closure;
};

class ValReturning : public Instruction  {
public:
    ValReturning(vector<MemLocation> readlocs, MemLocation writeloc, ValRetFun closure)
    : readlocs(readlocs), writeloc(writeloc), closure(closure) {}
    void fire(Interpreter*);
private:
    vector<MemLocation> readlocs;
    MemLocation writeloc;
    ValRetFun closure;
};


}
