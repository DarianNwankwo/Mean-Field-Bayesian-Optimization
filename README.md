# Bayesian Optimization with a Mean Field
## Abstract
Bayesian optimization (BO) is a powerful framework for optimizing expensive black-box functions. 
Standard BO methods rely on statistical surrogates to model the unknown objective and guide the 
sampling process through acquisition functions that balance exploration and exploitation. In 
this work, we extend BO to scenarios where the objective function’s mean field is partially or 
fully known, a setting that arises naturally in problems informed by prior scientific knowledge 
or physical models. By leveraging the known mean field, we construct surrogate models with 
enhanced fidelity, reducing uncertainty in regions aligned with the prior information. We 
explore the implications of this approach on acquisition function design, emphasizing the 
integration of prior mean knowledge into popular acquisition strategies such as Expected 
Improvement. We demonstrate the effectiveness of this framework through empirical evaluations on 
synthetic and real-world benchmarks, showcasing significant gains in optimization performance 
and sample efficiency.

## Considerations and Issues
<b>Minimizing the Negative Log Likelihood for the Hybrid Model</b>

The optimizer for learning the optimal hyperparameters seems to iterate beyond the point of marginal gains.
I need to investigate why, when I provided tolerances on the conditions for convergence, the optimizer
continues to seek gains.

<b>On Setting the Basis Functions and Kernel for the Hybrid Model</b>
We employ a constructor that preallocates our linear systems, however, we need to specify our kernel and polynomial
basis functions after this. Setting the kernel assumes the covariates and observations are already known, whereas
setting the polynomial basis functions does not. This may produce an error if not careful.

## Experimental Setup
The script will run evaluate a batch of test functions at a time. Now, we need to consider how we'll setup the remaining loops: trials, bo iterations, each surrogate model, and various acquisition functions.