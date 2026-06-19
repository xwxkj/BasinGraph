# CUTEst Discussion text

The CUTEst results provide an external nonlinear-optimization test beyond the BBOB suite and the internally constructed diagnostic functions. They also clarify the role of early termination. Multi-start L-BFGS-B and CMA-ES occasionally stopped before exhausting the nominal budget; this was not treated as a computational failure. For target-runtime calculations, an unsuccessful early termination was charged the full prescribed budget, whereas a successful early termination retained its observed time to target. This policy preserves the benefit of fast convergence without rewarding premature unsuccessful termination.

The dimension-stratified analysis should be used to identify where BasinGraph's basin-discovery mechanisms remain advantageous and where local or covariance-adaptive solvers dominate. The failure-mode table reports every problem on which BasinGraph did not attain the best median normalized residual, rather than suppressing unfavourable cases.
