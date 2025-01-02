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

## Software Design
