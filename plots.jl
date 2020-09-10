using DataFrames

include("cellType.jl")
include("utils.jl")

proteins = ["P20700", "P23246", "P49321", "Q96AE4", "O14979", "P14866", "Q92499", "P26599", 
"P62333", "P22626", "P31942", "P38159", "P09651", "Q15233", "P52272"]

using StatsPlots

# fibroblasts control
f_μ_c = [1., 1., 3., 5., 6., 5., 1., 1., 2., 1., 6., 10., 12., 13., 1.] # control means 
f_μ_s = [2., 3., 1., 3., 2., 4., 1., 3., 5., 5., 6., 6., 8., 5., 1.] # senescent means

f_σ_c = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
f_σ_s = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

# renal cells
r_μ_c = [5., 9., 8., 8., 3., 2., 1., 12., 11., 9., 5., 1., 3., 3., 3.]
r_μ_s = [4., 5., 3., 1., 3., 2., 1., 3., 9., 8., 5., 1., 3., 3., 3.]

r_σ_c = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
r_σ_s = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]


df = DataFrame(Proteins = proteins, 
                fibro_μ_c = f_μ_c, fibro_σ_c = f_σ_c, renal_μ_c = r_μ_c, renal_σ_c = r_σ_c,  
                fibro_μ_s = f_μ_s, fibro_σ_s = f_σ_s, renal_μ_s = r_μ_s, renal_σ_s = r_σ_s                 
                )

# plot expressions profile
p1 = bar(proteins, f_μ_c, yerr=f_σ_c, legend=false, xrotation=20, alpha=0.4, title="fibroblast", color="blue", titlefont=9, xaxis=false, yaxis=false)
p2 = bar(proteins, f_μ_s, yerr=f_σ_s, legend=false, xrotation=20, alpha=0.4, title="fibroblast", color="red", titlefont=9, xaxis=false, yaxis=false)
p3 = bar(proteins, r_μ_c, yerr=r_σ_c, legend=false, xrotation=20, alpha=0.4, title="renal", color="blue", titlefont=9, xaxis=false, yaxis=false)
p4 = bar(proteins, r_μ_s, yerr=r_σ_s, legend=false, xrotation=20, alpha=0.4, title="renal", color="red", titlefont=9, yaxis=false)
exp_plot = plot(p1, p2, p3, p4, layout = (4, 1), legend = false)

# load trained model
model = BSON.load("output/model_dirVAE_sparse.bson")

# find entropies of the latent space 
function latent_dist(input)
    alpha, scale = model[:Encoder](input)
    α  = exp.(alpha .* scale)
    Dirichlet(α)
end

f_c_l = latent_dist(f_μ_c)
f_s_l = latent_dist(f_μ_s)
r_c_l = latent_dist(r_μ_c)
r_s_l = latent_dist(r_μ_s)

entropies = Float64[]
labels = String[]
push!(entropies, entropy(f_c_l))
push!(labels, "quiescent")
push!(entropies, entropy(r_c_l))
push!(entropies, entropy(f_s_l))
push!(labels, "senescent")
push!(entropies, entropy(r_s_l))

ent_plot = plot(labels, reshape(entropies, (2,2))', label = ["fibro" "renal"], lw=3, legend=:bottomright)
