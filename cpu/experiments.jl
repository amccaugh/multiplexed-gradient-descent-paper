using ProgressMeter

include("./wmgd.jl")
include("./eval_mlp.jl")
include("./utilities.jl")
include("./datasets.jl")


function test_n_bit_parity_discrete(;
    n_bits,
    num_steps,
    num_pts_to_log,
    num_hidden,
    seed,
    eta,
    dtheta_type,
    n_perturb,
    norm_magnitude,
    dt,
    tau_x,
    tau_theta,
    accuracy_threshold,
    experiment_type,
    kwargs...,
)
# Setup data logging

log_every_n = floor(Int, num_steps/num_pts_to_log)
results_path = create_results_subfolder(experiment_type)

var_names_to_log = [
    "n",
    "t",
#     "theta",
#     "theta_ac",
    # "theta_ac_noisy",
#     "x",
#     "y_target",
#     "y",
#     "C",
#     "C_ac",
    # "G",
#     "G_int",
    # "G_true",
    # "angle",
    "accuracy",
    "mean_cost",
]

# Setup evaluation function
layer_sizes = [n_bits,num_hidden,1]
f_eval_config = mlp_config_generator(layer_sizes)
f_eval = f_mlp

# Setup cost
f_cost = MSE

# # Setup perturbation - FDMA


if dtheta_type == :tdma
    # Setup perturbation - TDMA
    f_dtheta_config = dtheta_tdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        )
    f_dtheta! = dtheta_tdma!
elseif dtheta_type == :cdma
    # Setup perturbation - CDMA
    f_dtheta_config = dtheta_cdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_cdma!
else
    error("Only CDMA and TDMA implemented right now")
end

# Setup dataset
dataset = generate_n_bit_parity_dataset(n_bits)
test_dataset = generate_n_bit_parity_dataset(n_bits)

logger = setup_logger(
    x_length = f_eval_config.x_length, y_length = f_eval_config.y_length,
    theta_length = f_eval_config.theta_length, num_steps = num_steps,
    log_every_n = log_every_n,
    var_names_to_log = var_names_to_log,
    )

rng = MersenneTwister(seed)
theta0 = rand(rng, Float64, f_eval_config.theta_length)/10

wmgd(
    f_eval = f_eval,
    f_eval_config = f_eval_config,
    f_cost = f_cost,
    f_dtheta! = f_dtheta!,
    f_dtheta_config = f_dtheta_config,
    logger = logger, 
    dataset = dataset, 
    test_dataset = test_dataset,
    theta = theta0,
    num_steps = num_steps,
    dt = dt,
    tau_x = tau_x,
    tau_hp = 1.6,
    tau_theta = tau_theta,
    eta = eta,
    analog = false,
    f_classification = threshold_accuracy_over_dataset,
    )


    # Setup parameters
    parameters_dict = Dict(
        :num_steps => num_steps,
        :tau_x => tau_x,
        :tau_theta => tau_theta,
        :eta => eta,
        :norm_magnitude => norm_magnitude,
        :seed => seed,
        :n => n_bits,
        :num_hidden => num_hidden,
        :n_perturb => n_perturb,
        :dt => dt,
        :accuracy_threshold => accuracy_threshold,
    )

    # Store any analysis results in a results dictionary
    # and append to results database
    results_dict = merge(
        Dict(:test_type => "$n_bits-bit parity"),
        analyze_final_accuracy_time(logger, accuracy_threshold),
        analyze_final_accuracy(logger),
        accuracy_factor_ten_progression(logger, num_steps),
        parameters_dict,
    )
    
    
    # Save any parameters of interest to results_path
    save_function_parameters(results_path, "parameters.txt"; results_dict...)
    plot_data(logger; results_path = results_path, title = experiment_type)
    close("all")
    GC.gc()
    return results_dict

end

function test_n_bit_parity(;
    n_bits,
    experiment_type,
    num_steps,
    num_pts_to_log,
    num_hidden,
    seed,
    eta,
    dtheta_type,
    n_perturb,
    norm_magnitude,
    dt,
    tau_x,
    tau_theta,
    is_analog = false,
    freq_start,
    freq_stop,
    accuracy_threshold = 0.9,
    kwargs...,
)
# Setup data logging

log_every_n = floor(Int, num_steps/num_pts_to_log)

var_names_to_log = [
    "n",
    "t",
#     "theta",
#     "theta_ac",
    # "theta_ac_noisy",
#     "x",
#     "y_target",
#     "y",
#     "C",
#     "C_ac",
    # "G",
#     "G_int",
    # "G_true",
    # "angle",
    "accuracy",
    "mean_cost",
]

# Setup evaluation function
layer_sizes = [n_bits,num_hidden,1]
f_eval_config = mlp_config_generator(layer_sizes)
f_eval = f_mlp
results_path = create_results_subfolder(experiment_type)

# Setup cost
f_cost = MSE

# # Setup perturbation - FDMA


if dtheta_type == :tdma
    # Setup perturbation - TDMA
    f_dtheta_config = dtheta_tdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_tdma!
elseif dtheta_type == :cdma
    # Setup perturbation - CDMA
    f_dtheta_config = dtheta_cdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_cdma!
elseif dtheta_type == :fdma
    f_dtheta_config = dtheta_fdma_config_generator(
        num_params = f_eval_config.theta_length,
        freq_start = freq_start,
        freq_stop = freq_stop,
        norm_magnitude = norm_magnitude,
        seed = seed,
    )
    f_dtheta! = dtheta_fdma!
else
    error("Only CDMA, FDMA and TDMA implemented right now")
end

# Setup dataset
dataset = generate_n_bit_parity_dataset(n_bits)
test_dataset = generate_n_bit_parity_dataset(n_bits)

logger = setup_logger(
    x_length = f_eval_config.x_length, y_length = f_eval_config.y_length,
    theta_length = f_eval_config.theta_length, num_steps = num_steps,
    log_every_n = log_every_n,
    var_names_to_log = var_names_to_log,
    )

rng = MersenneTwister(seed)
theta0 = rand(rng, Float64, f_eval_config.theta_length).-0.5
update_f_eval_config!(theta0, f_eval_config)
wmgd(
    f_eval = f_eval,
    f_eval_config = f_eval_config,
    f_cost = f_cost,
    f_dtheta! = f_dtheta!,
    f_dtheta_config = f_dtheta_config,
    logger = logger, 
    dataset = dataset, 
    test_dataset = test_dataset,
    theta = theta0,
    num_steps = num_steps,
    dt = dt,
    tau_x = tau_x,
    tau_hp = 1.6,
    tau_theta = tau_theta,
    eta = eta,
    analog = is_analog,
    f_classification = threshold_accuracy_over_dataset,
    )


    # Setup parameters
    parameters_dict = Dict(
        :num_steps => num_steps,
        :tau_x => tau_x,
        :tau_theta => tau_theta,
        :eta => eta,
        :norm_magnitude => norm_magnitude,
        :seed => seed,
        :n => n_bits,
        :num_hidden => num_hidden,
        :n_perturb => n_perturb,
        :dt => dt,
        :is_analog => is_analog,
        :dtheta_type => string(dtheta_type),
        :accuracy_threshold => accuracy_threshold,
    )

    # Store any analysis results in a results dictionary
    # and append to results database
    results_dict = merge(
        Dict(:test_type => "$n_bits-bit parity"),
        analyze_final_accuracy_time(logger, accuracy_threshold),
        analyze_final_accuracy(logger),
        accuracy_factor_ten_progression(logger, num_steps),
        parameters_dict,
    )
    
    
    # Save any parameters of interest to results_path
    save_function_parameters(results_path, "parameters.txt"; results_dict...)
    plot_data(logger; results_path = results_path, title = experiment_type)
    close("all")
    GC.gc()
    return results_dict

end


###################################################################

# NIST test 

###################################################################

function test_NIST(;
    num_steps,
    num_pts_to_log,
    num_hidden,
    seed,
    eta,
    dtheta_type,
    n_perturb,
    norm_magnitude,
    dt,
    tau_x,
    tau_theta,
    is_analog = false,
    kwargs...,
)
# Setup data logging

log_every_n = floor(Int, num_steps/num_pts_to_log)

var_names_to_log = [
    "n",
    "t",
     "theta",
#     "theta_ac",
    # "theta_ac_noisy",
#     "x",
#     "y_target",
#     "y",
#     "C",
#     "C_ac",
    # "G",
#     "G_int",
    # "G_true",
    # "angle",
    "mean_cost",
]

# Setup evaluation function
layer_sizes = [49,num_hidden,4]
f_eval_config = mlp_config_generator(layer_sizes)
f_eval = f_mlp

# Setup cost
f_cost = MSE

# # Setup perturbation - FDMA


if dtheta_type == :tdma
    # Setup perturbation - TDMA
    f_dtheta_config = dtheta_tdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_tdma!
elseif dtheta_type == :cdma
    # Setup perturbation - CDMA
    f_dtheta_config = dtheta_cdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_cdma!
elseif dtheta_type == :walsh
    # Setup perturbation - walsh
    f_dtheta_config = dtheta_walsh_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_walsh!
elseif dtheta_type == :fdma
    f_dtheta_config = dtheta_fdma_config_generator(
        num_params = f_eval_config.theta_length,
        freq_start = freq_start,
        freq_stop = freq_stop,
        norm_magnitude = norm_magnitude,
        seed = seed,
    )
    f_dtheta! = dtheta_fdma!
else
    error("Only CDMA, FDMA and TDMA implemented right now")
end

# Setup dataset
dataset = generate_NIST_dataset()
test_dataset = generate_NIST_dataset()


logger = setup_logger(
    x_length = f_eval_config.x_length, y_length = f_eval_config.y_length,
    theta_length = f_eval_config.theta_length, num_steps = num_steps,
    log_every_n = log_every_n,
    var_names_to_log = var_names_to_log,
    )

rng = MersenneTwister(seed)
theta0 = rand(rng, Float64, f_eval_config.theta_length).-0.5

wmgd(
    f_eval = f_eval,
    f_eval_config = f_eval_config,
    f_cost = f_cost,
    f_dtheta! = f_dtheta!,
    f_dtheta_config = f_dtheta_config,
    logger = logger, 
    dataset = dataset, 
    test_dataset = test_dataset,
    theta = theta0,
    num_steps = num_steps,
    dt = dt,
    tau_x = tau_x,
    tau_hp = 1.6,
    tau_theta = tau_theta,
    eta = eta,
    analog = is_analog,
    seed = seed,
    )

return logger



end

#############################
# test NIST angle
#############################

function test_NIST_angle(;
    num_steps,
    num_pts_to_log,
    num_hidden,
    seed,
    dtheta_type,
    n_perturb,
    norm_magnitude,
    dt,
    tau_x,
    is_analog = false,
    kwargs...,
    )
    # Setup data logging

    log_every_n = floor(Int, num_steps/num_pts_to_log)
    eta = 1.
    tau_theta = num_steps + 1.0

    var_names_to_log = [
    "n",
    "t",
    # "theta",
    #     "theta_ac",
    # "theta_ac_noisy",
    #     "x",
    #     "y_target",
    #     "y",
    #     "C",
    #     "C_ac",
    # "G",
    #     "G_int",
    # "G_true",
    "angle_full_dataset",
    #"mean_cost",
    ]

    # Setup evaluation function
    layer_sizes = [49,num_hidden,4]
    f_eval_config = mlp_config_generator(layer_sizes)
    f_eval = f_mlp

    # Setup cost
    f_cost = MSE

    # # Setup perturbation - FDMA


    if dtheta_type == :tdma
    # Setup perturbation - TDMA
    f_dtheta_config = dtheta_tdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_tdma!
    elseif dtheta_type == :cdma
    # Setup perturbation - CDMA
    f_dtheta_config = dtheta_cdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_cdma!
    elseif dtheta_type == :walsh
    # Setup perturbation - walsh
    f_dtheta_config = dtheta_walsh_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_walsh!
    elseif dtheta_type == :fdma
    f_dtheta_config = dtheta_fdma_config_generator(
        num_params = f_eval_config.theta_length,
        freq_start = freq_start,
        freq_stop = freq_stop,
        norm_magnitude = norm_magnitude,
        seed = seed,
    )
    f_dtheta! = dtheta_fdma!
    else
    error("Only CDMA, FDMA and TDMA implemented right now")
    end

    # Setup dataset
    dataset = generate_NIST_dataset()
    test_dataset = generate_NIST_dataset()


    logger = setup_logger(
    x_length = f_eval_config.x_length, y_length = f_eval_config.y_length,
    theta_length = f_eval_config.theta_length, num_steps = num_steps,
    log_every_n = log_every_n,
    var_names_to_log = var_names_to_log,
    )

    rng = MersenneTwister(seed)
    theta0 = rand(rng, Float64, f_eval_config.theta_length).-0.5

    wmgd(
    f_eval = f_eval,
    f_eval_config = f_eval_config,
    f_cost = f_cost,
    f_dtheta! = f_dtheta!,
    f_dtheta_config = f_dtheta_config,
    logger = logger, 
    dataset = dataset, 
    test_dataset = test_dataset,
    theta = theta0,
    num_steps = num_steps,
    dt = dt,
    tau_x = tau_x,
    tau_hp = 1.6,
    tau_theta = tau_theta,
    eta = eta,
    analog = is_analog,
    seed = seed,
    )
    angle = logger.angle_full_dataset.data
    angle = dropdims(angle, dims = 1)
    return angle

end

##############################

##########################################################
## Noisy NIST
###########################################################

function test_nist_noisy(;
    experiment_type = "test",
    num_steps = 100,
    num_pts_to_log = 100,
    num_hidden = 4,
    seed = 0,
    eta = 1.,
    dtheta_type = :cdma,
    n_perturb = 1,
    norm_magnitude = 0.01,
    dt = 1.,
    tau_x = 1.,
    tau_theta = 1.,
    is_analog = false,
    C_σ = 0.,
    theta_σ = 0.,
    sigmoid_σ = 0.,
    accuracy_threshold = 0.8,
    kwargs...,
)
# Setup data logging

log_every_n = floor(Int, num_steps/num_pts_to_log)
var_names_to_log = [
    "n",
    "t",
    #  "theta",
    #  "theta_ac",
    # "theta_ac_noisy",
    # "x",
    # "y_target",
    # "y",
    # "C",
    #  "C_ac",
    # "G",
#     "G_int",
    # "G_true",
    # "angle",
    "mean_cost",
    "accuracy",
]

# Setup evaluation function
layer_sizes = [49,num_hidden,4]
f_eval_config = noisy_mlp_config_generator(layer_sizes, sigmoid_σ, sigmoid_σ, sigmoid_σ, sigmoid_σ, seed)
f_eval = f_noisy_mlp

# Setup cost
f_cost = MSE

# # Setup perturbation - FDMA


if dtheta_type == :tdma
    # Setup perturbation - TDMA
    f_dtheta_config = dtheta_tdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_tdma!
elseif dtheta_type == :cdma
    # Setup perturbation - CDMA
    f_dtheta_config = dtheta_cdma_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        sparsity = 1,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_cdma!
elseif dtheta_type == :walsh
    # Setup perturbation - walsh
    f_dtheta_config = dtheta_walsh_config_generator(
                        n_perturb = n_perturb,
                        norm_magnitude = norm_magnitude,
                        num_params = f_eval_config.theta_length,
                        seed = seed,
                        )
    f_dtheta! = dtheta_walsh!
elseif dtheta_type == :fdma
    f_dtheta_config = dtheta_fdma_config_generator(
        num_params = f_eval_config.theta_length,
        freq_start = freq_start,
        freq_stop = freq_stop,
        norm_magnitude = norm_magnitude,
        seed = seed,
    )
    f_dtheta! = dtheta_fdma!
else
    error("Only CDMA, FDMA and TDMA implemented right now")
end

# Setup dataset
dataset = generate_NIST_dataset()
test_dataset = generate_NIST_dataset()

logger = setup_logger(
    x_length = f_eval_config.x_length, y_length = f_eval_config.y_length,
    theta_length = f_eval_config.theta_length, num_steps = num_steps,
    log_every_n = log_every_n,
    var_names_to_log = var_names_to_log,
    )

rng = MersenneTwister(seed)
theta0 = rand(rng, Float64, f_eval_config.theta_length).-0.5

wmgd(
    f_eval = f_eval,
    f_eval_config = f_eval_config,
    f_cost = f_cost,
    f_dtheta! = f_dtheta!,
    f_dtheta_config = f_dtheta_config,
    logger = logger, 
    dataset = dataset, 
    test_dataset = test_dataset,
    theta = theta0,
    num_steps = num_steps,
    dt = dt,
    tau_x = tau_x,
    tau_hp = 1.6,
    tau_theta = tau_theta,
    eta = eta,
    analog = is_analog,
    seed = seed,
    C_σ = C_σ,
    theta_σ = theta_σ,
    )

    results_path = create_results_subfolder(experiment_type) #put results folder creation here so they don't collide in time from distributed function

    # Setup parameters
    parameters_dict = Dict(
        :num_steps => num_steps,
        :tau_x => tau_x,
        :tau_theta => tau_theta,
        :eta => eta,
        :norm_magnitude => norm_magnitude,
        :seed => seed,
        :C_σ => C_σ,
        :theta_σ => theta_σ,
        :sigmoid_σ => sigmoid_σ,
        :num_hidden => num_hidden,
        :n_perturb => n_perturb,
        :dt => dt,
        :is_analog => is_analog,
        :accuracy_threshold => accuracy_threshold,
        :results_path => results_path,
    )

    # Store any analysis results in a results dictionary
    # and append to results database
    results_dict = merge(
        Dict(:test_type => "noisy_nist"),
        analyze_final_accuracy_time(logger, accuracy_threshold),
        analyze_final_accuracy(logger),
        accuracy_factor_ten_progression(logger, num_steps),
        parameters_dict,
    )
    
    
    # Save any parameters of interest to results_path
    save_function_parameters(results_path, "parameters.txt"; results_dict...)
    plot_data(logger; results_path = results_path, title = experiment_type)
    close("all")
    GC.gc()

    return results_dict
end
