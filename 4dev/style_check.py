#!/usr/bin/env python
import cgt
for (name,val) in cgt.__dict__.iteritems():
    if not name.startswith("_"):
        if not val.__doc__:
            print "API function %s requires docstring!"%name 