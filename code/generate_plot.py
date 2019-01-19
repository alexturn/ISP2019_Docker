import scipy.optimize as optimize
import numpy as np

from scipy.special import xlogy as xlx

from tqdm import tqdm

for T in tqdm([1,2,3,4,5]):

    def helper(nu, eps):

        def xlogx(x):
            return xlx(x,x)

        def entropy(x):
            return -xlogx(x) - xlogx(1-x)
        
        def entropy_grad(x):
            grad = np.zeros_like(x)
            grad = np.log(1 / x - 1)
            grad[x >= 1 - 1e-8] = +10
            grad[x <= 0 + 1e-8] = -10
            return grad

        def entropy_hess(x):
            hess = np.zeros_like(x)
            hess = -1 / ((1-x) * x)
            hess[x >= 1 - 1e-8] = -10
            hess[x <= 0 + 1e-8] = -10
            return hess

        def inner_func(mu, nu):
            return (entropy(mu - nu) - entropy(mu)).sum()

        def inner_jac(mu, nu):
            return entropy_grad(mu - nu) - entropy_grad(mu)

        def inner_hess(mu, nu):
            diag_elems = entropy_hess(mu - nu) - entropy_hess(mu)
            return np.diag(diag_elems)

        inner_function = lambda x: inner_func(x, nu)
        inner_jacobian = lambda x: inner_jac(x, nu)
        inner_hessian = lambda x: inner_hess(x, nu)

        A = np.concatenate([np.ones((1,T)),np.eye(T)],axis=0)
        lb = np.concatenate([np.array([(1 + eps) / 2. * T]), nu * np.ones(T)], axis=0)
        ub = np.concatenate([np.array([T]), np.ones(T)], axis=0)
        cons = optimize.LinearConstraint(A, lb, ub)
            
        result = optimize.minimize(fun=inner_function, 
                                jac=inner_jacobian,
                                constraints=cons,
                                hess=inner_hessian,
                                method="trust-constr",
                                x0=(1+eps) / 2 * np.ones(T),
                                options={'maxiter': 500, 'disp': False, 
                                'initial_constr_penalty': 10., 'initial_tr_radius': 1.})
        
        return result

    def grad_nu(x):
        grad = np.zeros_like(x)
        grad = np.log(x / (1 - x))
        grad[x >= 1 - 1e-8] = -10
        grad[x <= 0 + 1e-8] = +10
        return grad

    def hess_nu(x):
        hess = np.zeros_like(x)
        hess = -1 / ((1-x) * x)
        hess[x >= 1 - 1e-8] = -10
        hess[x <= 0 + 1e-8] = -10
        return hess


    def outer_func(nu, eps):
        result = helper(nu, eps)
        return -result.fun

    def outer_jac(nu, eps):
        result = helper(nu, eps)
        x = result.x
        return -grad_nu(x-nu).sum()

    def outer_hess(nu, eps):
        result = helper(nu, eps)
        x = result.x
        return -hess_nu(x - nu).sum()

    results = []
    x_results = []

    from tqdm import tqdm

    for epsilon in tqdm(np.arange(0, 1.1, 0.1)):

        outer_function = lambda nu: outer_func(nu, epsilon)
        outer_jacobian = lambda nu: outer_jac(nu, epsilon)
        outer_hessian = lambda nu: outer_hess(nu, epsilon)


        cons_constr = [optimize.LinearConstraint(np.eye(1), np.array([0.]), np.array([1.]))]

        result = optimize.minimize(fun=outer_function, 
                                jac=outer_jacobian,
                                constraints=cons_constr,
                                hess=outer_hessian,
                                method="trust-constr",
                                x0=np.array([epsilon / 2.]),
                                options={'maxiter': 1000, 'disp': True, 
                                'initial_constr_penalty': 10., 'initial_tr_radius': 4.})

        results += [-result.fun]
        x_results += [result.x]

    np.save('./runs_data/results_entropy_{}.npy'.format(T), np.array(results))