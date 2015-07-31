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
    virtual Tuple * run(Tuple *)=0;
    // called by instructions:
    virtual Object * get(MemLocation)=0;
    virtual void set(MemLocation, Object *)=0;
    virtual Object * getarg(int)=0;
    virtual ~Interpreter() {}
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
