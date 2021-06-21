This repo contains code to benchmark the inference of ODE models on graphs usingjulia and python. For inference in julia, [Turing](https://github.com/TuringLang/Turing.jl) will be used; for python, [numpyro](https://github.com/pyro-ppl/numpyro) will be used. 

The system used will be: 
<!-- $$
\frac{du}{dt} =  -\rho \sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{p}_j + \alpha \mathbf{p}_i(1-\mathbf{p}_i)
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bdu%7D%7Bdt%7D%20%3D%20%20-%5Crho%20%5Csum%5Climits_%7Bj%3D1%7D%5E%7BN%7D%5Cmathbf%7BL%7D_%7Bij%7D%5E%7B%5Comega%7D%5Cmathbf%7Bp%7D_j%20%2B%20%5Calpha%20%5Cmathbf%7Bp%7D_i(1-%5Cmathbf%7Bp%7D_i)"></div>
where $\mathbf{L}$ is the graph Laplacian given by: 
<!-- $$
\mathbf{L} = \mathbf{D} - \mathbf{A}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BL%7D%20%3D%20%5Cmathbf%7BD%7D%20-%20%5Cmathbf%7BA%7D"></div>
where $\mathbf{D}$ and $\mathbf{A}$ are the degree and adjacency matrices, respectively. 

The model is version of the Fisher-Kolmogorov–Petrovsky–Piskunov equation defined on a discete domain, namely a graph. In this case, we will use a Erdos-Renyi random graph of varying size and connection probability.