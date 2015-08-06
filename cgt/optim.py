import numpy as np, logging
from collections import deque
import cgt
import scipy.optimize as opt # maybe remove this dependency? only used for linesearch

class InverseHessianPairs(object):
    """
    LBFGS (inverse) Hessian approximation based on rotating list of pairs (step, delta gradient)
    that are assumed to approximately satisfy secant equation.
    """
    def __init__(self,max_num_pairs):
        self.syrhos = deque([],max_num_pairs) #pylint: disable=E1121
    def add(self,s,y):
        rho = 1./y.dot(s)
        self.syrhos.append((s,y,rho))
        if rho < 0: print "WARNING: rho < 0"
    def mvp(self,g):
        """
        Matrix-vector product        
        Nocedal & Wright Algorithm 7.4
        uses H0 = alpha*I, where alpha = <s,y>/<y,y>
        """
        assert len(self.syrhos) > 0

        q = g.copy()
        alphas = np.empty(len(self.syrhos))
        for (i,(s,y,rho)) in reversed(list(enumerate(self.syrhos))):
            alphas[i] = alpha = rho*s.dot(q)
            q -= alpha*y

        s,y,rho = self.syrhos[-1]
        ydoty = y.dot(y)
        sdoty = s.dot(y)
        gamma = sdoty/ydoty

        r = gamma*q

        for (i,(s,y,rho)) in enumerate(self.syrhos):
            beta = rho * y.dot(r)
            r += s * (alphas[i] - beta)

        return r

    
def lbfgs(f,fgrad,x0,maxiter=100,max_corr=25,grad_norm_tol=1e-9, ihp=None,ls_criteria="armijo"):
    """
    LBFGS algorithm as described by Nocedal & Wright in textbook Numerical Optimization
    """
    x = x0.copy()
    yield x
    if ihp is None: ihp = InverseHessianPairs(max_corr)
    oldg = fgrad(x)
    if ls_criteria=="armijo": fval = f(x)
    p = -oldg/np.linalg.norm(oldg)

    log = logging.getLogger("lbfgs")
    iter_count = 0
    while True:
        # TODO compare line searches
        g=None
        if ls_criteria == "strong_wolfe":
            alpha_star, _, _, fval, _, g = opt.line_search(f,fgrad,x,p,oldg)        
        elif ls_criteria == "armijo":
            import scipy.optimize.linesearch
            alpha_star,_,fval=scipy.optimize.linesearch.line_search_armijo(f,x,p,oldg,fval)
        else:
            raise NotImplementedError

        if alpha_star is None:
            log.error("lbfgs line search failed!")
            break
        s = alpha_star * p
        x += s
        yield x

        iter_count += 1
        
        if iter_count  >= maxiter:
            break

        if g is None: 
            log.debug("line search didn't give us a gradient. calculating")
            g = fgrad(x)

        if np.linalg.norm(g) < grad_norm_tol:
            break

        y = g - oldg
        ihp.add( s,y )
        p = ihp.mvp(-g)
        oldg = g

        log.info("lbfgs iter %i %8.3e",iter_count, fval)


def cg(f_Ax, b, cg_iters=10,callback=None,verbose=False,residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print titlestr%("iter","residual norm","soln norm")

    for i in xrange(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print fmtstr%(i, rdotr, np.linalg.norm(x))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr=newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print fmtstr%(i+1, rdotr, np.linalg.norm(x)) #pylint: disable=W0631
    return x

def preconditioned_cg(f_Ax, f_Minvx, b, cg_iters=10,callback=None,verbose=False,residual_tol=1e-10):
    """
    Demmel p 318
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = f_Minvx(b)
    y = p
    ydotr = y.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print titlestr%("iter","residual norm","soln norm")

    for i in xrange(cg_iters):
        if callback is not None:
            callback(x,f_Ax)
        if verbose: print fmtstr%(i, ydotr, np.linalg.norm(x))
        z = f_Ax(p)
        v = ydotr / p.dot(z)
        x += v*p
        r -= v*z
        y = f_Minvx(r)
        newydotr = y.dot(r)
        mu = newydotr/ydotr
        p = y + mu*p

        ydotr=newydotr

        if ydotr < residual_tol:
            break

    if verbose: print fmtstr%(cg_iters, ydotr, np.linalg.norm(x))

    return x



def test_cg():
    A = np.random.randn(5,5)    
    A = A.T.dot(A)
    b = np.random.randn(5)
    x = cg(lambda x: A.dot(x), b, cg_iters=5,verbose=True) #pylint: disable=W0108
    assert np.allclose(A.dot(x),b)


    x = preconditioned_cg(lambda x: A.dot(x), lambda x: np.linalg.solve(A,x), b, cg_iters=5,verbose=True) #pylint: disable=W0108
    assert np.allclose(A.dot(x),b)

    x = preconditioned_cg(lambda x: A.dot(x), lambda x: x/np.diag(A), b, cg_iters=5,verbose=True) #pylint: disable=W0108
    assert np.allclose(A.dot(x),b)



def lanczos(f_Ax, b, k):
    """
    Runs Lanczos algorithm to generate a orthogonal basis for the Krylov subspace
    b, Ab, A^2b, ...
    as well as the upper hessenberg matrix T = Q^T A Q

    from Demmel ch 6
    """

    assert k>1

    alphas = []
    betas = []
    qs = []

    q = b/np.linalg.norm(b)
    beta = 0
    qm = np.zeros_like(b)
    for j in xrange(k):
        qs.append(q)

        z = f_Ax(q)

        alpha = q.dot(z)
        alphas.append(alpha)
        z -= alpha*q + beta*qm

        beta = np.linalg.norm(z)
        betas.append(beta)

        print "beta",beta
        if beta < 1e-9:
            print "lanczos: early after %i/%i dimensions"%(j+1,k)
            break
        else:
            qm = q
            q = z/beta


    return np.array(qs,'float64').T, np.array(alphas,'float64'), np.array(betas[:-1],'float64')

def lanczos2(f_Ax, b, k,residual_thresh=1e-9):
    """
    More numerically stable but less efficient version
    """
    b = b.astype('float64')
    assert k>1
    H = np.zeros((k,k))
    qs = []

    q = b/np.linalg.norm(b)
    beta = 0

    for j in xrange(k):
        qs.append(q)

        z = f_Ax(q.astype(cgt.floatX)).astype('float64')
        for (i,q) in enumerate(qs):
            H[j,i] = H[i,j] = h = q.dot(z)
            z -= h*q

        beta = np.linalg.norm(z)
        if beta < residual_thresh:
            print "lanczos2: stopping early after %i/%i dimensions residual %f < %f"%(j+1,k,beta,residual_thresh)
            break
        else:
            q = z/beta
            
    return np.array(qs).T, H[:len(qs),:len(qs)]


def make_tridiagonal(alphas,betas):
    assert len(alphas)==len(betas)+1
    N = alphas.size
    out = np.zeros((N,N),cgt.floatX)
    out.flat[0:N**2:N+1] = alphas
    out.flat[1:N**2-N:N+1] = betas
    out.flat[N:N**2-1:N+1] = betas
    return out

def tridiagonal_eigenvalues(alphas,betas):
    T = make_tridiagonal(alphas,betas)
    return np.linalg.eigvalsh(T)

def test_lanczos():
    np.set_printoptions(precision=4)

    A = np.random.randn(5,5)    
    A = A.T.dot(A)
    b = np.random.randn(5)
    f_Ax = lambda x:A.dot(x) #pylint: disable=W0108
    Q,alphas,betas = lanczos(f_Ax,b,10)
    H = make_tridiagonal(alphas,betas)
    assert np.allclose( Q.T.dot(A).dot(Q), H)
    assert np.allclose(Q.dot(H).dot(Q.T),A)
    assert np.allclose(np.linalg.eigvalsh(H),np.linalg.eigvalsh(A))


    Q,H1 = lanczos2(f_Ax, b, 10)
    assert np.allclose(H,H1,atol=1e-6)


    print "ritz eigvals:"
    for i in xrange(1,6):
        Qi = Q[:,:i]
        Hi = Qi.T.dot(A).dot(Qi)
        print np.linalg.eigvalsh(Hi)[::-1]
    print "true eigvals:"
    print np.linalg.eigvalsh(A)[::-1]

    print "lanczos on ill-conditioned problem"
    A = np.diag(10**np.arange(5))
    Q,H1 = lanczos2(f_Ax, b, 10)
    print np.linalg.eigvalsh(H1)

    print "lanczos on ill-conditioned problem with noise"
    def f_Ax_noisy(x):
        return A.dot(x) + np.random.randn(x.size)*1e-3
    Q,H1 = lanczos2(f_Ax_noisy, b, 10)
    print np.linalg.eigvalsh(H1)

