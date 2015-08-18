import cgt
# X = cgt.matrix(fixed_shape=(10,3))
y = cgt.vector(fixed_shape=(3,))
w = cgt.vector(fixed_shape=(5,))
# z = X.dot(y)
y+w
# cgt.print_tree(cgt.core.simplify(cgt.shape(z)))