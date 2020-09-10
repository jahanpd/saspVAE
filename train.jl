using Flux
using ProgressMeter: Progress, next!
using JLD
using BSON


include("cellType.jl")
include("dataLoader.jl")
include("loss.jl")
include("VAE.jl")

# set parameters
η = 1e-3                # learning rate
λ = 0.001f0              # regularization paramater
β = 2f0                 # beta weight for the KLD term
batch_size = 32        # batch size
epochs = 100            # number of epochs
warm_up = 500     

train_loader = load_data(batch_size)

x_dims = size(train_loader.data)[1]
hidden_dims = 40
z_dims = 8

# define location to save trained model
save_path = "output"    # results path
!ispath(save_path) && mkpath(save_path)

# build networks
Encoder = encoder(x_dims, x_dims, z_dims, 2, true)
Decoder = decoder(z_dims, hidden_dims, x_dims, 2)

# set optimiser
opt = ADAM(η)

# params of networks
θ_vae = Flux.params(
    Encoder.α_z,Encoder.scale, Encoder.layer, Encoder.layers,
    Decoder.out, Decoder.layer, Decoder.layers,
)

global history = Any[]
counter = 0
for epoch in 1:epochs
    @info "Epoch $(epoch)"
    progress = Progress(length(train_loader))

    global counter
    global debug = 0
    local record = Array{Float32}(undef, 4, 0) # store training losses

    local loss, recon, kld, reg # create local loss variables

    counter < warm_up ? no_noise = 0f0 : no_noise = 1f0 # if burning in reduces the noise in sampling to 0
    for x in train_loader #for x in [train_loader.data[:,1:batch_size]] #
        loss, back = Flux.pullback(θ_vae) do

            α_z, scale  = Encoder(x)

            remapping = exp.(α_z.*scale)

            z = sampling(remapping, no_noise)

            x_ = Decoder(z)


            kld = KLD(remapping)

            recon = mean( sum( (x .- x_).^2, dims=1 ) ) # mean squared error

            reg = λ * sum(x->sum(abs.(x)), Flux.params(Encoder.α_z, Encoder.layer, Encoder.layers))


            counter < warm_up ? total = recon : total = recon + β*(kld) + reg

            total

        end
        # backprop gradients
        ∇ = back(1f0)
        Flux.Optimise.update!(opt, θ_vae, ∇)

        # record data
        record = hcat(record, [loss, recon, kld, reg])

        # determine smoothing step
        size(record)[2] > 10 ? idx = 10 : idx = size(record)[2] - 1

        # progress meter
        next!(progress; showvalues=[(:loss, mean(record[1,end-idx:end])),
                                    (:recon, mean(record[2,end-idx:end])),
                                    (:kld, mean(record[3,end-idx:end])),
                                    (:reg, mean(record[4,end-idx:end])),
                                    (:counter, counter),
                                    (:debug, typeof(loss)),
                                     ])

        counter += 1
    end
    push!(history, record)
end

model_path = joinpath(save_path, "model_dirVAE_sparse.bson")
let Encoder = Encoder, Decoder = Decoder
    BSON.@save model_path Encoder Decoder
    @info "Model saved: $(model_path)"
end
train_path = joinpath(save_path, "history_sparse.jld")
save(train_path, "history", history)
