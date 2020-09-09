import odl
import numpy as np

class regularization_methods():
    
    def __init__(self, forward_operator):
        self.operator = forward_operator

        
    def TV(self, x0, data, alpha=0.01, niter=100, xtrue = None):
        gradient = odl.Gradient(self.operator.domain)
        op = odl.BroadcastOperator(self.operator, gradient)
        f = odl.solvers.ZeroFunctional(op.domain)
        l2 = 0.5 * odl.solvers.L2NormSquared(self.operator.range).translated(data)
        TV = alpha * odl.solvers.GroupL1Norm(gradient.range, 2)
        g = odl.solvers.SeparableSum(l2, TV)
        
        # Stepsize estimation
        op_norm = 1.1 * odl.power_method_opnorm(op)
        tau = 1.0 / op_norm  # Step size for the primal variable
        sigma = 1.0 / op_norm  # Step size for the dual variable
        
        err = []
        rel_err = []

        def err_append(xi):
            err.append(TV(gradient(xi)) + l2(self.operator(xi)))
            pass

        cb = err_append

        if not xtrue is None:
            def rel_err_append(xi):

                rel_err.append(np.linalg.norm(xi - xtrue)/np.linalg.norm(xtrue))
                pass

            cb = lambda xi: (err_append(xi), rel_err_append(xi))
        
        x = op.domain.element(x0[:]) # to copy the x0
        odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=cb)
        return x, err, rel_err
    
    def Wavelet(self, x0, data, alpha=0.01, niter=100, nlevels=5, wavelet='haar', xtrue=None):
        W = odl.trafos.WaveletTransform(self.operator.domain, wavelet=wavelet, 
                                        nlevels=nlevels)

        Wtrafoinv = W.inverse
        
        l1 = alpha*odl.solvers.L1Norm(Wtrafoinv.domain)
        l2 = odl.solvers.L2NormSquared(self.operator.range).translated(data)
        
        data_discr = 0.5 * l2 * self.operator * Wtrafoinv
        
        opnorm = ((self.operator * Wtrafoinv).norm(estimate=True))**2
        stepsize = 1./(2*opnorm)
            
        x = W(x0)

        err = []
        rel_err = []

        def err_append(xi):
            err.append(l1(xi) + data_discr(xi))
            pass

        cb = err_append

        if not xtrue is None:
            def rel_err_append(xi):
                z = Wtrafoinv(xi)
                rel_err.append(np.linalg.norm(z - xtrue)/np.linalg.norm(xtrue))
                pass

            cb = lambda xi: (err_append(xi), rel_err_append(xi))
    
        odl.solvers.accelerated_proximal_gradient(x, f=l1, g=data_discr, niter=niter, gamma=stepsize, callback=cb)
        return Wtrafoinv(x), err, rel_err
