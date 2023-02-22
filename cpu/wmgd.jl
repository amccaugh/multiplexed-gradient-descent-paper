##### Major TODO
# Add Digits dataset

##### Minor TODO
# Write parameter_combination/dataframe/boxplot code
# Optimize CDMA/TDMA
# MIMO: spatial multiplexing to create multiple Cost values (one for each batch element, or one per each section)

##### MAYBE
# Write gaussian-distributed CDMA
# Write brownian-motion-wandering CDMA
# Add noise to cost
# Gate `G` so that when tau_x / tau_theta changes, error isn't accumulated

using Dates

include("./utilities.jl")
include("./perturb.jl")
include("./eval_mlp.jl")
include("./logger.jl")
include("./plotting.jl")
include("./datasets.jl")

function wmgd(;
    f_eval::Function,
    f_eval_config::eval_config,
    f_cost::Function,
    f_dtheta!::Function,
    f_dtheta_config::dtheta_config,
    logger::DataLogger,
    dataset::Dataset,
    test_dataset::Dataset,
    theta::Vector{Float64},
    num_steps::Int,
    dt::Float64,
    tau_x::Float64,
    tau_hp::Float64,
    tau_theta::Float64,
    eta::Float64,
    analog::Bool,
    seed::Int = 0,
    dataset_shuffle::Bool = true,
    C_σ::Float64 = 0.,
    theta_σ = 0.,
    f_classification = nothing,
    )

    # Intializiation
    start_time = time()
    num_params::Int = length(theta)
    tau_theta_next = tau_theta
    C::Float64 = 0.0
    C0::Float64 = 0.0
    C_ac::Float64 = 0.0
    C_ac_prev::Float64 = 0.0
    C_prev::Float64 = 0.0
    G::Vector{Float64} = zeros(num_params)
    G_int::Vector{Float64} = zeros(num_params)
    G_int_prev::Vector{Float64} = zeros(num_params)
    theta_ac::Vector{Float64} = zeros(num_params)
    theta_total::Vector{Float64} = zeros(num_params)
    t::Float64 = 0.0
    tau_x_next::Float64 = 0
    is_x_changed::Bool = true
    is_theta_updated::Bool = true
    dataset_idx::Int = 0
    compute_angle::Bool = logger.angle.enabled
    compute_angle_full_dataset::Bool = logger.angle_full_dataset.enabled
    compute_mean_cost::Bool = logger.mean_cost.enabled
    finite_difference_amplitude::Float64 = 0.001
    rng::AbstractRNG = MersenneTwister(seed)
    dataset_order::Vector{Int} = collect(1:dataset.num_samples)

    # Temporary static initialization (to be removed)
    x::Vector{Float64} = zeros(Float64, f_eval_config.x_length)*NaN
    y::Vector{Float64} = zeros(Float64, f_eval_config.y_length)*NaN
    y_target::Vector{Float64} = zeros(Float64, f_eval_config.y_length)*NaN
    theta_ac_noisy::Vector{Float64} = copy(theta)
    G_true::Vector{Float64} = copy(theta)
    mean_cost::Float64 = NaN
    accuracy::Float64 = NaN
    angle::Float64 = 0.0
    angle_full_dataset::Float64 = 0.0

    for n in 1:num_steps
        is_logged_timestep = (((n-1) % logger.log_every_n) == 0)

 # Compute mean cost and angle only if required by logger
        if  ((compute_mean_cost == true) && (analog == true) && is_logged_timestep) || 
            ((compute_mean_cost == true) && (analog == false) && is_theta_updated && is_logged_timestep)
            mean_cost = mean_cost_over_dataset(test_dataset, theta, f_eval, f_eval_config, f_cost)
            if isnothing(f_classification)
                accuracy = classification_accuracy_over_dataset(test_dataset, theta, f_eval, f_eval_config)
            else 
                accuracy = f_classification(test_dataset, theta, f_eval, f_eval_config)
            end
            is_theta_updated = false
        end

        #### If more than tau_x has elapsed, update input/target data
        if t >= tau_x_next
            tau_x_next += tau_x
            if dataset_shuffle == true
                if dataset_idx % dataset.num_samples == 0
                    shuffle!(rng, dataset_order)
                end
            end

            shuffled_data_idx = dataset_order[dataset_idx % dataset.num_samples + 1]
            get_training_batch!(x, y_target, shuffled_data_idx, dataset)
            dataset_idx += 1
            is_x_changed = true
        end

        # Update weights if more than tau_theta has elapsed, or doing lowpass-style weight updates
        if (t >= tau_theta_next) || (analog == true)
            # Update θ parameters/weights
            theta .-= eta.*G_int .+ randn(rng, Float64)*theta_σ
            if analog == false
                # Reset the integrated gradient
                G_int .*= 0
            end
            tau_theta_next += tau_theta

            is_theta_updated = true
        end

        #### Update parameters
        # Update the theta_ac perturbation
        f_dtheta!(theta_ac, t, n, f_dtheta_config)
        # Combine current theta + perturbation (without using memory)
        theta_total .= theta .+ theta_ac #  0.007us / 0 allocations per

        #### Compute y output and the cost
        y = f_eval(x, theta_total, f_eval_config) # 0.3us / 6 allocations per
        C = f_cost(y, y_target) + randn(rng, Float64)*C_σ # 0.15us / 1 allocations per
        
        #### Calculate C_ac
        if analog
            # Highpass filter the cost using https://en.wikipedia.org/wiki/High-pass_filter#Discrete-time_realization
            if n==1 || is_x_changed 
                C_prev = C
            end
            C_ac = tau_hp/(tau_hp + dt)*C_ac_prev + tau_hp/(tau_hp + dt)*(C - C_prev)
        else
            # Filter the cost by subtracting a constant offset C0
            if is_x_changed || is_theta_updated
                # Recompute C0
                C0 = f_cost(f_eval(x, theta, f_eval_config), y_target)
            end
            C_ac = C-C0
        end

        #### Compute G, the psuedo-gradient
        G .= C_ac*dt.*theta_ac./(f_dtheta_config.norm_magnitude^2)

        #### Compute G_int: integrate G either discretely or lowpass-style
        if analog
            if n==0 # Correct initialization
                G_int_prev .= G_int
            end
            G_int .= G.*(dt/(tau_theta + dt)) .+ G_int_prev.*(tau_theta/(tau_theta + dt))
            G_int_prev .= G_int
        else
            G_int .+= G
        end

        if compute_angle == true && is_logged_timestep
            G_true = finite_difference_gradient(f_eval, f_eval_config, f_cost,
                                                theta, x, y_target, finite_difference_amplitude)
            angle = _angle_between(G_int, G_true)
        end
        if compute_angle_full_dataset == true && is_logged_timestep
            G_true = finite_difference_gradient_full_dataset(f_eval, f_eval_config, f_cost,
                                                theta, dataset, finite_difference_amplitude)
            angle_full_dataset = _angle_between(G_int, G_true)
        end


        log_data(
            logger,
            n,
            t,
            theta,
            theta_ac,
            theta_ac_noisy,
            x,
            y_target,
            y,
            C,
            C_ac,
            G,
            G_int,
            G_true,
            angle,
            angle_full_dataset,
            mean_cost,
            accuracy,
            )

        # Store the previous cost values for next highpass computation
        C_prev = C
        C_ac_prev = C_ac
        t += dt
        is_x_changed = false
    end

    return theta
end