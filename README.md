This repo contains code to benchmark the inference of ODE models on graphs usingjulia and python. For inference in julia, [Turing](https://github.com/TuringLang/Turing.jl) will be used; for python, [numpyro](https://github.com/pyro-ppl/numpyro) will be used. 

The system used will be: 
<!-- $$
\frac{du}{dt} =  -\rho \sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{p}_j + \alpha \mathbf{p}_i(1-\mathbf{p}_i)
$$ --> 

<div align="center"><img style="background: white;" src="svg/8rfkyjIXz8.svg"></div>
where $\mathbf{L}$ is the graph Laplacian given by: 
<!-- $$
\mathbf{L} = \mathbf{D} - \mathbf{A}
$$ --> 

<div align="center"><img style="background: white;" src="svg/qU10S8mfhi.svg"></div>
where $\mathbf{D}$ and $\mathbf{A}$ are the degree and adjacency matrices, respectively. 

The model is version of the Fisher-Kolmogorov–Petrovsky–Piskunov equation defined on a discete domain, namely a graph. In this case, we will use a Erdos-Renyi random graph of varying size and connection probability.