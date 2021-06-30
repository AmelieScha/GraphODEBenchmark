using DelimitedFiles
using DifferentialEquations
using Turing
using StaticArrays
using Plots


const SL = readdlm("graphs/ER-10-05.csv", ',', Float64) |> SMatrix{10,10}

function NetworkFKPP(du, u, p, t)
    du .= -p[1] * SL * u .+ p[2] .* u .* (1 .- u)
end

u0 = zeros(10)
u0[5] = 0.1

p = [0.5, 1.5]

t_span = (0.0,10.0)

prob = ODEProblem(NetworkFKPP, u0, t_span, p)
sol = solve(prob, Tsit5(), saveat=1.0);

@model function fit(data, prob)
    σ ~ InverseGamma(2, 3)

    k ~ truncated(Normal(1.0,2.0), 0, Inf)
    a ~ truncated(Normal(1.0,2.0), 0, Inf)

    problem = remake(prob, p=[k, a])
    predictions = solve(problem, Tsit5(), saveat=1.0)

    for i in 1:10
        data[:,i] .~ MvNormal(predictions[i], σ)
    end
end

@model function fit1(data, prob)
    σ ~ InverseGamma(2, 3)

    k ~ truncated(Normal(1.0,2.0), 0, Inf)
    a ~ truncated(Normal(1.0,2.0), 0, Inf)

    problem = remake(prob, p=[k, a])
    predictions = solve(problem, Tsit5(), saveat=1.0)
    println(size(data))
    println(size(predictions))
    data .~ arraydist(Normal.(predictions, σ))
end

@model function fit2(data, prob)
    σ ~ InverseGamma(2, 3)

    k ~ truncated(Normal(1.0,2.0), 0, Inf)
    a ~ truncated(Normal(1.0,2.0), 0, Inf)

    problem = remake(prob, p=[k, a])
    predictions = solve(problem, Tsit5(), saveat=1.0)

    Turing.@addlogprob! sum(loglikelihood.(Normal.(predictions,σ), data)) 
end

m = fit(Array(sol), prob);
chain = sample(m, NUTS(0.65), 2_000)

m1 = fit1(Array(sol), prob);
chain1 = sample(m1, NUTS(0.65), 2_000)

m2 = fit2(Array(sol), prob);
chain1 = sample(m2, NUTS(0.65), 2_000)


