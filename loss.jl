using LinearAlgebra
using SpecialFunctions

lnΓ = loggamma
diΓ = digamma

### probability functions

# function KLD(α, β; agg=mean, ϵ=eps(Float32))
#     @assert size(α)[1] == size(β)[1]
#     α0, β0 = sum(α.+ϵ, dims=1), sum(β.+ϵ, dims=1)
#     @. $agg(lnΓ(α0) - $sum(lnΓ(α.+ϵ), dims=1) - lnΓ(β0) + $sum(lnΓ(β.+ϵ), dims=1) + $sum( (α-β .+ ϵ)*(diΓ(α.+ϵ) - diΓ(α0.+ϵ)) , dims=1))
# end


function KLD(α; agg=mean, ϵ=eps(Float32))
    α0 = sum(α.+ϵ, dims=1)
    @. $abs( $agg( lnΓ(α0)
            - $sum(lnΓ(α.+ϵ), dims=1)
            + $sum( α * diΓ(α .+ ϵ), dims=1)
            - $sum( α * diΓ(α0 .+ ϵ), dims=1)
            - $mean( diΓ(α .+ ϵ) - diΓ(α0 .+ ϵ), dims=1) ) )
end