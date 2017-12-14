#!/usr/bin/env python
import cgt
for (name,val) in cgt.__dict__.items():
    if not name.startswith("_"):
        if not val.__doc__:
            print("API function %s requires docstring!"%name) 


for (name,val) in cgt.core.__dict__.items():
    if isinstance(val, type) and issubclass(val, cgt.core.Op):
        if val.get_native_compile_info == cgt.core.Op.get_native_compile_info:
            print("Op %s is missing 'get_native_compile_info'!"%name)

