# Online Bayesian Projection Pursuit Regression

A machine learning approach suited to the following context:
* You have no lack of inspiration for generating input signals, but you're not sure which (combinations of them) have the most predictive power
* You are at least pretty sure your millions of signals are orthogonal
* You are also confident the response you're targeting is nonlinear in the inputs and it might be impossible to linearize them
* Part of that nonlinearity owes itself to the locality of useful parts of the input space: an awful lot of it is uninformative but when *this guy* is high and *that guy* is low, profit. 
* Please, please, not another neural net. 
* Your signal-to-noise ratio is probably low. You've been known to break out in high-fives at an $R^2$ of 0.021
* You reserve the right to access the posterior prediction distribution, not just the single prediction point that may or may not have some colossal variance it's not telling you about. 
* You'd find it particularly handy if you didn't have to retrain on all that history every time some new data comes in
* Come to think of it, you'd quite like it if you could just set and forget, and let it carry on merrily without your micromanagement. 
* While there might be some interaction between signals, it's likely to be simple. You're not seeking an effect that looks like a high order polynomial over many of your inputs, more like the additive effect of multiple local-but-low-order terms involving just a couple of signals at once
* You prize robustness, but you love pure speed. 

[Collins et al](https://arxiv.org/abs/2210.09181) formulated a Bayesian approach to Projection Pursuit Regression and while attractive, [their code](https://github.com/gqcollins/BayesPPR) is more research- than production-driven. [This implementation](./src/bppr.cpp) takes the algorithm they came up with, strips out some of its scope, introduces incremental online updates and does as much of the heavy lifting as possible in cholesky-space, obviating lots of matrix decompositions and inversions. In C++. 

BPPR uses the Metropolis Hastings algorithm for accepting or rejecting proposals to change the family of ridge functions that fit the response. MCMC methods are often regarded like a boilersuit: Gets the job done but ugh, could you not have dressed up (your intractable normalizing integral) *just a little*? The technique suits projection pursuit however, giving up little in efficiency to other inevitably iterative approaches, while retaining the robustness that MCMC is reknowned for.  

## The Build

BPPR depends **heavily** on the Intel Math Kernel Library, so you'll need to be within cooey of an Intel processor and have installed the MKL. You will also need the [constexpr](https://github.com/elbeno/constexpr) project for compile-time math utilities in `cx_math.h` as well as the low-level logging library [L3](https://github.com/undoio/l3). *C++20* required. See [Build Instructions](./doc/build.md) for more information. 

## Do The Work

Setting up the training and learning is a three-step process:
1. Crystallizing the compile-time parameters that describe your data as well as your BPPR expectations and hopes, get the uint64 encoding for those parameters. 
2. Make an initial guess at a BPPR state and write a prescribed binary file combining that with your training data, naming it after the encoding. Insert the encoding into `bppr.cpp`, double check the data path and recompile. 
3. Run the executable (or library) to perform the training and online updating / prediction against out-of-sample data. 

See [Run Instructions](./doc/run.md) for more information. 

## A Toy Example

Setting $X_{1}=\sin(t)+\varepsilon, X_{2}=\cos(t)+\varepsilon, X_{3}=\tanh(X_{1}+X_{2})+\varepsilon$ and $y=\alpha_{1}X_{1}+\alpha_{2}X_{2}+\alpha_{3}sgn(X_{3})e^{-X_{3}^{2}}+\epsilon$, we have a nonlinear, almost bang-bang type control we're trying to model with variance greatest where the control switches sign. 

See [Example](./doc/example.md) for training results and a look at the posterior distribution.

## On the Shoulders

Several disfigurements of Collins' work have been made which reduces the applicability of this implementation:
* Only continuous (no categorical) inputs and you're responsible for your own input scaling and avoidance of unit roots
* Only the Zellner-Siow prior for the regression coefficients, no flat prior
* Enforced adaptation of the number-of-interacting-terms weights, and the prior signal weights themselves. 
* No direct shifting up of the zero'th quantile to concentrate focus on one end of the projection's spectrum

But there are also some innovations:
* Addition of an Anti-Similarity Measure to the ratio of priors when Bayes-considering an adjustment such as a new addition to the ridge function family. A major drawback of the ability to conjure up a new projection is that if you test one that is similar to one you already have, you've introduced multicolinearity. Regressions *love* to exploit multicolinearity by promising a magnificent fit to the data, just to sucker-punch you with crazy predictions from new data when you were feeling all good. Without some resistance to this tendency, the ridge functions will gravitate to one another and generate spurious results. By balancing any promised improvement to the sse with how similar the projection is to any of those already used, we can make more robust choices about accepting new proposals. We use the [Joint Generalized Cosine Similarity](https://arxiv.org/abs/2505.03532) to achieve this. 
* Rather than sampling a zero'th quantile to test locality/concentration of projection and response, we 'sigmoidize' the raw quantiles with a parameter that makes them bulge to the extremes or compress to the median. The regression coefficients themselves would then be relied on to pick up any asymmetry between high and low projection responses, so you still get concentration but in a perhaps more rounded way. 
* New proposals to the ridge function lineup, as well as new data samples, are propagated through to their effect on the cholesky matrices themselves where the updates are actually made. This saves on a huge amount of computation. 

------

### We need to talk about RFPF

The [Rectangular Full Packed Format](https://arxiv.org/abs/0901.1696) is an ingenious way of using only the minimum required storage for triangular matrices while still enabling use of standard BLAS and LAPACK routines. Traditionally one has used the full `NxN` allocation, almost half of which is ignored. BLAS and LAPACK are pretty good about only using what's there - they won't busywork themselves into multiplying by zeros if they're told - but you still have that uneasy shadow of inefficiency hanging over you. It just seems so... *wasteful*. Hence the decision to utilize RFPF. 

Which was soon regretted utterly. The format adds significant complexity to the update procedure for proposals, ridge function modifications in particular, and the wasted storage otherwise is barely on the radar of the average machine learning practitioner. Only truly large-scale learning problems would benefit from such storage hawkishness, and at that scale you would probably have more pressing problems, such as assumed `unsigned short` variables no longer able to hold the lengths you're driving them to. 

We saw it through regardless, in the hope that a heart-wrenchingly resource-constrained problem out there will benefit. Perhaps a smart hydronic manifold that needs to forecast heating demand based on expected temperature, humidity and the amount of TV you'll be watching, running on an old-school Raspberry Pi. If that's what you're putting together, let us know so we can experience some thin relief. 