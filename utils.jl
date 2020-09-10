using Statistics
using DataFrames
using Random
using LinearAlgebra
using Flux.Zygote: Buffer, ignore

###### tools to standardise data
mutable struct standardScaler
    colnames
    μ_
    σ_
end

function fit(D::DataFrame, cols::Array, scaler::standardScaler)
    means = []
    stds = []
    for name in cols
        append!(means, mean(D[:,name]))
        append!(stds, std(D[:,name]))
    end
    scaler(cols, means, stds)
end


function fit_transform!(D::Array, cols::Array)
    means = []
    stds = []
    for (i, name) in enumerate(cols)
        append!(means, mean(D[:,i]))
        append!(stds, std(D[:,i]))
    end

    scaler = standardScaler(cols, means, stds)

    for (index, name) in enumerate(scaler.colnames)
        D[:,index] = (D[:,index] .- scaler.μ_[index]) ./ scaler.σ_[index]
    end

    scaler
end

function invert!(D::DataFrame, scaler::standardScaler)
    for (index, name) in enumerate(scaler.colnames)
        D[:,name] = (D[:,name] .* scaler.σ_[index]) .+ scaler.μ_[index]
    end
end

function invert_array!(D::Array, scaler::standardScaler)
    for (index, name) in enumerate(scaler.colnames)
        D[index,:] = (D[index,:] .* scaler.σ_[index]) .+ scaler.μ_[index]
    end
end


##### train test split

function train_test_split(x::DataFrame, y=0; frac = 0.3)
    D = convert(Array{Float32}, x)
    rows, cols = size(D)
    idx = shuffle(1:rows)
    if y==0
        return D[idx[floor(Int, 0.3*rows)+1:end],:],  D[idx[1:floor(Int, 0.3*rows)],:]
    else
        y = convert(Array{Float32}, y)
        return D[idx[floor(Int, 0.3*rows)+1:end],:],  D[idx[1:floor(Int, 0.3*rows)],:], y[idx[floor(Int, 0.3*rows)+1:end],:],  y[idx[1:floor(Int, 0.3*rows)],:]
    end
end


#### sampling

function sampling(α, modifier=1f0)
    # Sampling Reparameterization using Laplace Apprixmation between k-dimensional Dirichlet and Logistic Normal
    μ, σ_ = α_μσ(α)
    softmax(μ .+ randn(size(σ_)).*σ_.*modifier)
end

function α_μσ(α, ϵ = eps(Float32))
    K, b = size(α)
    μ = log.(α .+ ϵ) .- mean(log.(α .+ ϵ), dims=1)
    σ_ = ((1 - 2/K) .* (1 ./ (α .+ ϵ))) .+ ((1/K^2).* sum((1 ./ (α .+ ϵ)) , dims=1))
    μ, σ_
end