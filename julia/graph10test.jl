using DelimitedFiles
using DifferentialEquations
using Turing
using StaticArrays
using Plots

const SL = readdlm("graphs/ER-100-05.csv", ',', Float64)
const L = readdlm("graphs/ER-100-05.csv", ',', Float64)

function NetworkFKPP(du, u, p, t)
    du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
end


u0 = zeros(100)
u0[50] = 0.1

p = [0.05, 1.5]

t_span = (0.0,10.0)

prob = ODEProblem(NetworkFKPP, u0, t_span, p)
sol = solve(prob, Tsit5(), saveat=1.0);

plot(sol)

@model function fit(data, prob)
    σ ~ InverseGamma(2, 3)

    k ~ truncated(Normal(), 0.0, 10.0)
    a ~ truncated(Normal(), 0.0, 10.0)

    problem = remake(prob, p=[k, a])
    predictions = solve(problem, Tsit5(), saveat=1.0)
    for i in 1:11
        data[:,i] .~ MvNormal(predictions[:,i], σ)
    end
end

@model function fit1(data, prob)
    σ ~ InverseGamma(2, 3)


    k ~ truncated(Normal(), 0.0, 10.0)
    a ~ truncated(Normal(), 0.0, 10.0)

    problem = remake(prob, p=[k, a])
    predictions = solve(problem, Tsit5(), saveat=1.0)

    data .~ arraydist(Normal.(Array(predictions), σ))
end

@model function fit2(data, prob)
    σ ~ InverseGamma(2, 3)

    k ~ truncated(Normal(), 0.0, 10.0)
    a ~ truncated(Normal(), 0.0, 10.0)

    problem = remake(prob, p=[k, a])
    predictions = solve(problem, Tsit5(), saveat=1.0)

    Turing.@addlogprob! sum(loglikelihood.(Normal.(predictions,σ), data)) 
end

using BenchmarkTools

m = fit(Array(sol), prob);
chain = sample(m, NUTS(0.65), 2_000)

m1 = fit1(Array(sol), prob);
@benchmark chain1 = sample(m1, NUTS(0.65), 2_000)

m2 = fit2(Array(sol), prob);
@benchmark chain2 = sample(m2, NUTS(0.65), 2_000)

using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
Turing.emptyrdcache()