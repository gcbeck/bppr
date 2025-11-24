# BPPR - Run Instructions

BPPR almost always allocates on the stack, eschews STL containers in favour of POD arrays and mostly passes those arrays by reference, not pointer. The miniscule efficiency gain is secondary to the enforced array length expectations - even if the occasional `reinterpret_cast` is then sheepishly called for - which sidesteps some buggy territory, and keeps the door open to a lock-free future. The data directory is also hardcoded - we're incorrigibly holding out for compile-time initialization from file - and how often do you *really* switch it anyway?

The upshot is an initial compilation whenever you're setting up a different learning problem. Here's what you'll need to have decided on before you can run anything:
* `N` The number of training samples in your in-sample set. `N ≤ 65535`.
* `M` The number of signals in the training set. `M ≤ 127`.
* `R` The maximum number of ridge functions. `R ≤ 127` but if `R >> M` then you will just get lots of birth proposals aborted because the projection cholesky matrix will be close to singular - that is, it's hard to find another ridge direction orthogonal to the ones you already have. For technical reasons however it can be helpful to allow `R` a bit of freedom, so perhaps keep `R ≤ M+2` in mind. 
* `A` The maximum number of signals that can interact in a single ridge function. `A ≤ 3`.
* `K` The number of knots, or degree of localization in each projection. Higher has you cosying up to variance in the bias-variance dilemma. `1 ≤ K ≤ 63` but something around 4 is recommended. 
* `P` The number of samples making up the posterior distribution. `P ≤ 65535` but ensure that it is small enough or `N` large enough that $N \ge P\bar{r}$ where $\bar{r}$ is the average number of ridge functions over all of the elements in the posterior set. 
* `I` The flag indicating whether to include an intercept term. `I ∈ {0,1}`. **Note** `I=1` has not yet been adequately tested. 

You retrieve the *encoding* by concatenating these parameters, in that order, as an argument to `./bppr -e `. For example:
```
./bppr -e 256,3,8,2,4,16,0
```

------

Armed with the encoding, write a binary file `<encoding>.bppr` that observes the BPPR persistence protocol encapsulating both your training set and a guess at an initial BPPR state. Follow the `bpprfile` parts of [this synthetic data example](./tst/TestData.py) closely to get started. If you have no particular priors on how you want your initial state influenced, just work from
```
r = 1
nu = [0] 
chat = [1, 0, ..., 0] # Length M
ihat = [0.0, 1/M, 2/M, ..., 1.0, M]  # Length M+2  
ahat = [0.0, 1/A, 2/A, ..., 1.0, A]    # Length A+2
that = 0.98
```
Figure out how to insert your training data `(X,y)` in there and you'll be away. 

------

Finally, check the data path `constexpr auto REPO` and the encoding `constexpr auto T` in [bppr.cpp](./src/bppr.cpp) that they match where you've generated the `.bppr` file and your encoding respectively. Compile and run, concatenating the desired number of MCMC iterations you want in the Burn and Adapt phases, and the period for posterior caching; for example:
```
./bppr -d 256,16,4
```
Adding argument `-w` will write the updated BPPR state to your data directory, allowing for chaining of state updates over successive bppr runs so BPPR is automatically initialized with the latest data on launch. 

------

BPPR is most useful as a library, integrated into external code that calls `predict` and `update` functions. However for demonstration purposes and perhaps some utility, [the executable](./src/bppr.cpp) will pull in out-of-sample data from an `.updt` file and process the samples successively as if performing the updates online.  

------

Logs are written in a binary look-up form to `<encoding>.l3b`. Produce a human-readable translation with 
```
l3log /path/to/encoding.l3b /path/to/executable/bppr
```
assuming the `l3log` function in `~/.bashrc` given in the [build instructions](./build.md)
