##
include("./utilities.jl")
using Random
using Distributions

abstract type eval_config end

""" Configuration / internal state for f_mlp """
struct mlp_config <: eval_config
    layer_sizes::Vector{Int}
    weight_matrices::Vector{Matrix{Float64}}
    bias_vectors::Vector{Vector{Float64}}
    activations::Vector{Vector{Float64}}
    x_length::Int
    y_length::Int
    theta_length::Int
end

""" Generates an initial configuration / internal state for f_mlp """
function mlp_config_generator(layer_sizes::Vector{Int})
    weight_matrices = create_weight_matrices(layer_sizes::Vector{Int})
    bias_vectors = create_bias_vectors(layer_sizes)
    activations = [zeros(Float64,ls) for ls in layer_sizes]
    x_length = layer_sizes[1]
    y_length = layer_sizes[end]
    theta_length = 0
    theta_length += sum([length(w) for w in weight_matrices])
    theta_length += sum([length(b) for b in bias_vectors])
    config = mlp_config(layer_sizes, weight_matrices, bias_vectors, activations,
                        x_length, y_length, theta_length)
    
    return config
end

""" Updates the MLP `config` object so that the values of the weight
matrices/bias vectors used in the MLP calculation are replaced by those in
theta.  Operates in a way that no new memory is allocated -- only the existing
memory inside config is updated """
function update_f_eval_config!(theta::Vector{Float64}, config::mlp_config)
    idx::Int = 0
    # For each weight matrix `w` stored in `config`, replace the values of `w`
    # with values from `theta`
    for w in config.weight_matrices
        for i in eachindex(w)
            w[i] = theta[idx+i]
        end
        idx += length(w)
    end

    # For each bias vector `b` stored in `config`, replace the values of `b`
    # with values from `theta`
    for b in config.bias_vectors
        for i in eachindex(b)
            b[i] = theta[idx+i]
        end
        idx += length(b)
    end

end


# config = mlp_config_generator([2,2,1])
# theta = Float64[1:9;]
# update_f_eval_config!(theta, config)

""" MLP evaluation function. Weight matrices and bias vectors are stored inside
`config` and used for evaluation with `x` """
function f_mlp(x::Vector{Float64}, config::mlp_config)
    W_list = config.weight_matrices
    b_list = config.bias_vectors
    a_list = config.activations

    # Run MLP
    a_list[1] .= x
    for n in 1:length(W_list)
        # a = a_list[n]
        w = W_list[n]
        b = b_list[n]
        # Perform w*a
        mul!(a_list[n+1], w, a_list[n])
        # Subtract b
        a_list[n+1] .-= b
        # Apply nonlinearity
        a_list[n+1] .= sigmoid.(a_list[n+1])
    end
    return a_list[end]
end




""" MLP evaluation function. Takes a vector `theta` and in-place updates the
weights/biases inside the MLP `config` struct, then performs the evaluation """
function f_mlp(x::Vector{Float64}, theta::Vector{Float64}, config::mlp_config)
    update_f_eval_config!(theta, config)
    return f_mlp(x, config)
end

#############################################################
# Noisy MLP: In this case, there 
# is some change to the activation 
# functions of individual neuron from the designed sigmoid
##############################################################

""" Configuration / internal state for f_mlp """

struct noisy_mlp_config <: eval_config
    layer_sizes::Vector{Int}
    weight_matrices::Vector{Matrix{Float64}}
    bias_vectors::Vector{Vector{Float64}}
    activations::Vector{Vector{Float64}}
    x_length::Int
    y_length::Int
    theta_length::Int
    x0::Vector{Vector{Float64}}
    y0::Vector{Vector{Float64}} 
    k::Vector{Vector{Float64}}
    L::Vector{Vector{Float64}}
end

""" Generates an initial configuration / internal state for f_mlp """
function noisy_mlp_config_generator(layer_sizes::Vector{Int},
    midpoint_x0_std::Float64, asymmetry_y0_std::Float64, 
    steepness_k_std::Float64, amplitude_L_std::Float64,
    seed::Int
    )

    rng = MersenneTwister(seed)
    
    weight_matrices = create_weight_matrices(layer_sizes::Vector{Int})
    bias_vectors = create_bias_vectors(layer_sizes)
    activations = [zeros(Float64,ls) for ls in layer_sizes]

    x0 = [randn(rng, Float64, ls)*midpoint_x0_std for ls in layer_sizes[2:end]]
    y0 = [randn(rng, Float64, ls)*asymmetry_y0_std for ls in layer_sizes[2:end]]
    k = [randn(rng, Float64, ls)*steepness_k_std .+ 1. for ls in layer_sizes[2:end]]
    L = [randn(rng, Float64, ls)*amplitude_L_std .+ 1. for ls in layer_sizes[2:end]]

    x_length = layer_sizes[1]
    y_length = layer_sizes[end]
    theta_length = 0
    theta_length += sum([length(w) for w in weight_matrices])
    theta_length += sum([length(b) for b in bias_vectors])
    config = noisy_mlp_config(layer_sizes, weight_matrices, bias_vectors, activations,
                        x_length, y_length, theta_length, 
                        x0, y0, k, L)
    
    return config
end

""" Updates the MLP `config` object so that the values of the weight
matrices/bias vectors used in the MLP calculation are replaced by those in
theta.  Operates in a way that no new memory is allocated -- only the existing
memory inside config is updated """
function update_f_eval_config!(theta::Vector{Float64}, config::noisy_mlp_config)
    idx::Int = 0
    # For each weight matrix `w` stored in `config`, replace the values of `w`
    # with values from `theta`
    for w in config.weight_matrices
        for i in eachindex(w)
            w[i] = theta[idx+i]
        end
        idx += length(w)
    end

    # For each bias vector `b` stored in `config`, replace the values of `b`
    # with values from `theta`
    for b in config.bias_vectors
        for i in eachindex(b)
            b[i] = theta[idx+i]
        end
        idx += length(b)
    end

end


# config = mlp_config_generator([2,2,1])
# theta = Float64[1:9;]
# update_f_eval_config!(theta, config)

""" MLP evaluation function. Weight matrices and bias vectors are stored inside
`config` and used for evaluation with `x` """
function f_noisy_mlp(x::Vector{Float64}, config::noisy_mlp_config)
    W_list = config.weight_matrices
    b_list = config.bias_vectors
    a_list = config.activations
    x0 = config.x0 
    y0 = config.y0
    k = config.k
    L = config.L

    # Run MLP
    a_list[1] .= x
    for n in 1:length(W_list)
        # a = a_list[n]
        w = W_list[n]
        b = b_list[n]
        # Perform w*a
        mul!(a_list[n+1], w, a_list[n])
        # Subtract b
        a_list[n+1] .-= b
        # Apply nonlinearity
        a_list[n+1] .= logistic.(a_list[n+1], x0[n], y0[n], k[n], L[n])
    end
    return a_list[end]
end

""" MLP evaluation function. Takes a vector `theta` and in-place updates the
weights/biases inside the MLP `config` struct, then performs the evaluation """
function f_noisy_mlp(x::Vector{Float64}, theta::Vector{Float64}, config::noisy_mlp_config)
    update_f_eval_config!(theta, config)
    return f_noisy_mlp(x, config)
end

# # ## Main loop
# # x = Float64[1,2]
# # theta = Float64[1:9;]
# x = [1.30400005, 0.94708096]
# theta = [ 0.08890469, -0.09341224,  0.4528472 ,  0.07417558, -0.43737221, 0.29524113, 0.,  0.,  0.]
# layer_sizes = Int[2,2,1]
# seed = 0
# f_eval_config = noisy_mlp_config_generator([2,2,1], 0.01, 0.01, 0.01, 0.01, seed)

# f_noisy_mlp(x, f_eval_config)
# #@btime f_mlp(x, theta, config)

