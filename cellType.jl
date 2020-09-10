using LinearAlgebra
using Distributions

mutable struct proteome
    names::Array{String, 1} # name of the protein in the proteome
    μ::Array{Float64, 1}
    σ::Array{Float64, 1}
    d::MvNormal
    function proteome(names, μ, σ)
        new(
            names,
            μ,
            σ,
            MvNormal(μ, σ)
        )
    end
end

function sample(cell::proteome, n::Int)
    rand(cell.d, n)
end

function plot_cell(cell::proteome)
    d = Normal[]
    for (i, protein) in enumerate(cell.names)
        push!( d, Normal(cell.μ[i], cell.σ[i]) )
    end
    plot(d, fill=0, alpha=0.4)
end