# define module for vanilla vae
using Flux

### https://arxiv.org/pdf/1901.02739.pdf
### https://github.com/jxnl/dirichlet-vae/blob/master/vae/dirichlet.py

###### create encoder
struct encoder
    layer
    layers
    α_z
    scale
    hidden
    encoder(input_dim, hidden_dim, latent_dim, layers, hidden=true) = new(
        Dense(input_dim, hidden_dim, relu),  # latent space means
        [Dense(hidden_dim, hidden_dim, relu) for n in 1:layers],
        Dense(hidden_dim, latent_dim, tanh),  # latent space means
        Dense(hidden_dim, latent_dim, softplus),  # log std of the prior
        hidden
    )
end

function (enc::encoder)(x)
    h = enc.layer(x) # will skip hidden layer 1 if hidden is false

    if length(enc.layers) > 0
        layers(x) = foldl((x, m) -> m(x), enc.layers, init = x)
        h = layers(h)
    end

    if enc.hidden
        α_z = enc.α_z(h)
        scale = enc.scale(h)
        α_z, scale
    else
        α_z = enc.α_z(x)
        scale = enc.scale(x)
        α_z, scale
    end
end

###### create decoder
struct decoder
    layer
    layers
    out
    decoder(latent_dim, hidden_dim, output_dims, layers) = new(
        Dense(latent_dim, hidden_dim, relu),  # output cont means
        [Dense(hidden_dim, hidden_dim, relu) for n in 1:layers],
        Dense(hidden_dim, output_dims),  # output cont means
    )
end

function (dec::decoder)(z)
    h = dec.layer(z)
    if length(dec.layers) > 0
        layers(x) = foldl((x, m) -> m(x), dec.layers, init = x)
        h = layers(h)
    end
    dec.out(Dropout(0.2)(h))
end