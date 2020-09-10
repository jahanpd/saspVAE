using Flux.Data: DataLoader
using Random

include("cellType.jl")
include("utils.jl")

proteins = ["P20700", "P23246", "P49321", "Q96AE4", "O14979", "P14866", "Q92499", "P26599", 
"P62333", "P22626", "P31942", "P38159", "P09651", "Q15233", "P52272"]

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

function load_data(batch_size=32; nsample=1000)
    fibro_c = proteome(proteins, f_μ_c, f_σ_c)
    renal_c = proteome(proteins, r_μ_c, r_σ_c)
    fibro_s = proteome(proteins, f_μ_s, f_σ_s)
    renal_s = proteome(proteins,r_μ_s, r_σ_s)

    fc = sample(fibro_c, nsample)
    rc = sample(renal_c, nsample)
    fs = sample(fibro_s, nsample)
    rs = sample(renal_s, nsample)

    x = hcat(fc, rc, fs, rs)

    idx = shuffle(1:size(x)[2])

    scaler = fit_transform!(x, proteins)
    
    train_loader = DataLoader(x[:, idx], batchsize=batch_size)

    return train_loader
end