using Random
using BenchmarkTools
using Hadamard

abstract type dtheta_config end

""" Configuration / internal state for dtheta_fdma """
struct dtheta_fdma_config <: dtheta_config
    freqs::Vector{Float64}
    phase_offsets::Vector{Float64}
    norm_magnitude::Float64
    amplitude::Float64
end

""" Creates a "config" struct for dtheta_fdma that holds all the state needed to
run dtheta_fdma (e.g. frequencies, magnitude, etc) """
function dtheta_fdma_config_generator(;
    num_params::Int,
    freq_start::Float64,
    freq_stop::Float64,
    norm_magnitude::Float64,
    seed::Int,
)
    rng = MersenneTwister(seed)
    freqs = collect(range(freq_start, freq_stop, length = num_params))
    shuffle!(rng, freqs)
    phase_offsets = 2π*rand(rng, num_params)
    amplitude = norm_magnitude/sqrt(num_params)*sqrt(2)
    
    config = dtheta_fdma_config(
        freqs,
        phase_offsets,
        norm_magnitude,
        amplitude
    )
    return config
end

# function dtheta_fdma(t::Real, config::dtheta_fdma_config)
#     freqs = config.freqs
#     phase_offsets = config.phase_offsets 
#     magnitude = config.magnitude
#     perturbation = sin.(2π*freqs*t + phase_offsets)*magnitude
#     return perturbation
# end

""" Updates the perturbation value theta_ac to its new value.  The operation is
performed in-place so no new memory is required """
function dtheta_fdma!(theta_ac::Vector{Float64}, t::Float64, n::Int, config::dtheta_fdma_config)
    freqs = config.freqs
    phase_offsets = config.phase_offsets 
    amplitude = config.amplitude
    for m in eachindex(theta_ac)
        theta_ac[m] = sin(2π*freqs[m]*t+ phase_offsets[m])*amplitude #
    end
end



# # Example usage
# config = dtheta_fdma_config_generator(
#     num_params = 9,
#     freq_start = 1.0,
#     freq_stop = 1.5,
#     magnitude = 0.01,
#     seed = 0,
# )

# theta_ac = Float64[1:9;]
# t = 0
# println(theta_ac)
# dtheta_fdma!(theta_ac, t, config)
# println(theta_ac)



################### TDMA ##########################
""" Configuration / internal state for dtheta_tdma """

struct dtheta_tdma_config <: dtheta_config
    n_perturb::Int
    norm_magnitude::Float64
    sparsity::Int
    num_params::Int
    n_order::Vector{Int} # order in which parameters are varied
end

""" Creates a "config" struct for dtheta_tdma that holds all the state needed to
run dtheta_fdma (e.g. frequencies, norm_magnitude, etc) """
function dtheta_tdma_config_generator(;
    n_perturb::Int,
    norm_magnitude::Float64,
    sparsity::Int,
    num_params::Int,
    seed::Int,
)
    
    n_order = 1:num_params
    rng = MersenneTwister(seed)
    n_order = shuffle(rng, n_order)

    config = dtheta_tdma_config(
        n_perturb,
        norm_magnitude,
        sparsity,
        num_params,
        n_order,
    )
    return config
end

""" Updates the perturbation value theta_ac to its new value.  The operation is
performed in-place so no new memory is required """

function dtheta_tdma!(theta_ac::Vector{Float64}, t::Float64, n::Int, config::dtheta_tdma_config)
    
    n_perturb = config.n_perturb
    sparsity = config.sparsity
    norm_magnitude = config.norm_magnitude
    num_params = config.num_params
    n_slot = n_perturb * sparsity
    n_order = config.n_order

    for i in 1:num_params
        theta_ac[i] = 0
    end

    if n % n_slot < n_perturb
        n_slot = (floor(Int, (n / n_perturb)) % num_params)
        i = n_order[n_slot+1]
        theta_ac[i] = norm_magnitude
    end
end

################### CDMA - random ##########################

""" Configuration / internal state for dtheta_cdma """

mutable struct dtheta_cdma_config <: dtheta_config
    n_perturb::Int
    norm_magnitude::Float64
    sparsity_vec::Vector
    num_params::Int
    amplitude::Float64
    seed::Int
    rng::AbstractRNG
    n_slot::Int
end

""" Creates a "config" struct for dtheta_cdma that holds all the state needed to
run dtheta_fdma (e.g. frequencies, norm_magnitude, etc) """
function dtheta_cdma_config_generator(;
    n_perturb::Int,
    norm_magnitude::Float64,
    sparsity::Int,
    num_params::Int,
    seed::Int,
    )

    rng = MersenneTwister(seed)
    amplitude = sparsity*norm_magnitude/sqrt(num_params)
    added_sparsity = zeros(2*(sparsity-1))
    amplitude_vals = [amplitude, -1*amplitude]
    sparsity_vec = [amplitude_vals; added_sparsity]
    n_slot = -1
    config = dtheta_cdma_config(
        n_perturb,
        norm_magnitude,
        sparsity_vec,
        num_params,
        amplitude,
        seed,
        rng,
        n_slot,
    )
    return config
end

""" Updates the perturbation value theta_ac to its new value.  The operation is
performed in-place so no new memory is required """

function dtheta_cdma!(theta_ac::Vector{Float64}, t::Float64, n::Int, config::dtheta_cdma_config)
    
    n_perturb = config.n_perturb
    sparsity_vec = config.sparsity_vec
    amplitude = config.amplitude
    num_params = config.num_params
    n_slot_old = config.n_slot
    rng = config.rng

    n_slot = floor(Int, (n / n_perturb))

    if n_slot > n_slot_old
        theta_ac .= rand(rng, sparsity_vec, num_params)
        config.n_slot += 1
    end
end

############# Walsh code / Hadamard matrices ################

""" Configuration / internal state for dtheta_walsh """

struct dtheta_walsh_config <: dtheta_config
    n_perturb::Int
    norm_magnitude::Float64
    num_params::Int
    amplitude::Float64
    code::Matrix{Int8}
end

""" Creates a "config" struct for dtheta_cdma that holds all the state needed to
run dtheta_fdma (e.g. frequencies, norm_magnitude, etc) """
function dtheta_walsh_config_generator(;
    n_perturb::Int,
    norm_magnitude::Float64,
    num_params::Int,
    seed::Int
    )
    rng = MersenneTwister(seed)
    amplitude = norm_magnitude/sqrt(num_params)
    size_hadamard = ceil(Int, num_params/4)*4
    code = Hadamard.hadamard(size_hadamard)
    code = code[shuffle(1:end), :]
    config = dtheta_walsh_config(
        n_perturb,
        norm_magnitude,
        num_params,
        amplitude,
        code,
    )
    return config
end

""" Updates the perturbation value theta_ac to its new value.  The operation is
performed in-place so no new memory is required """

function dtheta_walsh!(theta_ac::Vector{Float64}, t::Float64, n::Int, config::dtheta_walsh_config)
    
    n_perturb = config.n_perturb
    amplitude = config.amplitude
    num_params = config.num_params
    code = config.code

    n_slot = floor(Int, (n / n_perturb))
    code_number = (n_slot % (size(code)[1]) ) 
    
    if floor(Int, n_slot/(size(code)[1])) % 2 == 0
        multiplier = -1.
    else
        multiplier = 1.
    end

    for i in 1:length(theta_ac)
        theta_ac[i] = code[i, code_number+1] * amplitude * multiplier
    end    

end


# #Example usage
# config = dtheta_walsh_config_generator(
#     n_perturb = 1,
#     norm_magnitude = 0.01,
#     num_params = 9,
# )

# theta_ac = Float64[1:9;]
# n = 7
# t = 17.
# println(theta_ac)
# @btime dtheta_walsh!(theta_ac, t, n, config)
# println(theta_ac)


# config = dtheta_cdma_config_generator(
#     n_perturb = 1,
#     norm_magnitude = 0.1,
#     sparsity = 1,
#     num_params = 9,
#     seed = 0,
# )

# theta_ac = Float64[1:9;]
# t = 0.5
# println(theta_ac)
# @btime dtheta_cdma!(theta_ac, t, n, config)
# println(theta_ac)