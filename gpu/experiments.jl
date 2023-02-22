
include("./wmgd_gpu.jl")
include("./utilities.jl")
include("./analysis.jl")
using Random
using Flux
using BenchmarkTools
# using TOML
# using BSON: @save, @load #package for saving and loading flux networks
using Dates

"""
Functions in this folder are given the following:
    - A directory/path called results_path to dump results into
    - Parameters/arguments for running a single WMGD/backprop run
They are responsible for
    - Execute a single run of a given experiment (e.g. CIFAR10 wmgd)
    - Plotting data and saving the figure
    - Saving the function inputs which were used in a text file (e.g. tau_x = 1)
    - Saving the logger as a jld2
    - Performing analysis and returning the results in a `results_dict`
"""

######################
# FashionMNIST
######################

function experiment_FashionMNIST(;
    results_path,
    num_steps = 5000,
    batch_size = 1000,
    eta = 10,
    tau_p = 1,
    tau_x = 1,
    tau_theta = 1,
    label_smoothing = 0.01,
    seed = 0,
    norm_magnitude = 0.1,
    cost_type = "crossentropy",
    C_ac_limit = Inf,
    convergence_accuracy_threshold = 0.9,
    starting_params = "test.jld2"
    )
    #### Parameter setup
    Random.seed!(seed)
    f_eval_cpu = Chain(
        Conv((3,3), 1=>16, relu),
        MaxPool((2,2)),
        Conv((3,3), 16=>32, relu),
        MaxPool((2,2)),
        Conv((3,3), 32=>32, relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(32, 10),
    )
    
    
    f_eval = f_eval_cpu |> gpu # Prevents hanging on to the previous run's parameters

    # Setup f_cost and accuracy functions
    f_ce(y, y_target) = Flux.Losses.logitcrossentropy(y, y_target)
    f_mse(y, y_target) = Flux.Losses.mse(y, y_target) 
    if cost_type == "crossentropy"
        f_cost = f_ce
    elseif cost_type == "mse"
        f_cost = f_mse
    else
        error("Invalid cost_type")
    end
    accuracy(x, y) = mean(onecold(f_eval(x), 1:10) .== onecold(y, 1:10))


    # Select dataset
    Xtrain, Ytrain, Xtest, Ytest = get_fashion_mnist_dataset(test_samples = 1000,
                                                        batch_size = batch_size,
                                                        label_smoothing = label_smoothing)

    logger = Dict()
    logger = wmgd_gpu(;
        f_eval = f_eval,
        f_cost = f_cost,
        f_dtheta! = f_dtheta!,
        accuracy = accuracy,
        num_steps = num_steps,
        logger = logger,
        tau_p = tau_p,
        tau_x = tau_x,
        tau_theta = tau_theta,
        eta = eta,
        norm_magnitude = norm_magnitude,
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = Xtest,
        Ytest = Ytest,
        seed = seed,
        C_ac_limit = C_ac_limit,
        )

    # Plot things and save to results_path
    title1 = "FashionMNIST using WMGD"
    title2 = "τ_θ = $tau_theta / τ_x = $tau_x / norm_mag = $norm_magnitude\n"
    title3 = "batch_size = $batch_size / label_smoothing = $label_smoothing\n"
    plot_logger(logger, 
        title = title1*title2*title3, 
        filename = "_wmgd_plot.png", 
        to_plot = nothing,
        path = results_path,
        )
    
    # Setup parameters
    parameters_dict = Dict(
        :batch_size => batch_size,
        :label_smoothing => label_smoothing,
        :num_steps => num_steps,
        :tau_x => tau_x,
        :tau_theta => tau_theta,
        :tau_p => tau_p,
        :eta => eta,
        :norm_magnitude => norm_magnitude,
        :seed => seed,
        :C_ac_limit => C_ac_limit,
        :cost_type => cost_type,
    )

    # Store any analysis results in a results dictionary
    # and append to results database
    results_dict = merge(
        Dict(:test_type => "fashion_mnist_wmgd"),
        analyze_solution_convergence_time(logger, convergence_accuracy_threshold),
        analyze_final_accuracy(logger),
        analyze_max_accuracy(logger),
        analyze_wallclock_time(logger),
        accuracy_factor_ten_progression(logger),
        parameters_dict,
    )
    append_to_results_database("results_database.csv", results_dict)

    # Save any parameters of interest to results_path
    save_function_parameters(results_path, "parameters.txt"; parameters_dict...)

    # Save logger
    save_logger(logger, results_path, "logger.jld2")

    CUDA.reclaim()
    GC.gc(true)
    return results_dict
end


# ##########
# # FashionMNIST backprop
# ##########
function experiment_FashionMNIST_backprop(;
    results_path,
    seed = 0,
    batch_size = 1000,
    label_smoothing = 0.01,
    starting_params = nothing,
    num_backprop_epochs = 1000,
    eta = 1,
    convergence_accuracy_threshold = Inf)
    
    #### Parameter setup
    #### Parameter setup
    Random.seed!(seed)
    f_eval_cpu = Chain(
        Conv((3,3), 1=>16, relu),
        MaxPool((2,2)),
        Conv((3,3), 16=>32, relu),
        MaxPool((2,2)),
        Conv((3,3), 32=>32, relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(32, 10),
    )

    #### Choose optimizer
    opt = Flux.Optimise.Descent(eta) # η

    #### Setup f_cost and accuracy functions

    f_eval_backprop = f_eval_cpu |> gpu

    if !isnothing(starting_params)
        load_parameters(starting_params, f_eval_backprop)
    end

    f_cost_backprop(x, y) = Flux.Losses.logitcrossentropy(f_eval_backprop(x), y)
    accuracy_backprop(x, y) = mean(onecold(f_eval_backprop(x), 1:10) .== onecold(y, 1:10))

    #### Select dataset
    label_smoothing = label_smoothing
    Xtrain, Ytrain, Xtest, Ytest = get_fashion_mnist_dataset(test_samples = batch_size,
                                                        batch_size = batch_size,
                                                        label_smoothing = label_smoothing)

    backprop_logger = backprop(;
        num_epochs = num_backprop_epochs,
        stopaccuracy = convergence_accuracy_threshold,
        f_eval = f_eval_backprop,
        f_cost = f_cost_backprop,
        accuracy = accuracy_backprop,
        opt,
        Xtrain,
        Ytrain,
        Xtest,
        Ytest  
    )

    # Save plots to results_path 
    title1 = "FashionMNIST using backprop\n"
    title2 = "eta = $eta / batch_size = $batch_size / label_smoothing = $label_smoothing\n"
    plot_logger(backprop_logger; title=title1*title2, filename = "backprop_plot.png", path = results_path)
    save_logger(backprop_logger, results_path, "backprop_logger.jld2")

    convergence_time = backprop_logger["iteration"][end]
    final_accuracy = analyze_final_accuracy(backprop_logger)
    wallclock_time = analyze_wallclock_time(backprop_logger)
    max_accuracy = analyze_max_accuracy(backprop_logger)
    # Store any analysis results in a results dictionary

    results_dict = Dict()
    results_dict[:convergence_time] = convergence_time
    results_dict[:final_accuracy] = final_accuracy
    results_dict[:wallclock_time] = wallclock_time
    results_dict[:max_accuracy] = max_accuracy
    results_dict[:test_type] = "FashionMNIST_backprop"

    # Bookkeeping - write parameters to file in results folder for convenienec
    save_function_parameters(results_path, 
        seed = seed,
        batch_size = batch_size,
        eta = eta,
        label_smoothing = label_smoothing,
        num_backprop_epochs = num_backprop_epochs,
        convergence_time = convergence_time,
        final_accuracy = final_accuracy,
        wallclock_time = wallclock_time,
        max_accuracy = max_accuracy,
        )

    CUDA.reclaim()
    GC.gc(true)
    return results_dict
end

######################
# CIFAR10
######################

function experiment_CIFAR10_wmgd(;
    results_path,
    eta, 
    num_steps = 1000,
    tau_p = 1,
    tau_x = 1,
    tau_theta = 1,
    seed = 0,
    norm_magnitude = 0.001,
    batch_size = 1000,
    label_smoothing = 0.01,
    starting_params = nothing,
    cost_type = "crossentropy",
    C_ac_limit = 0.01,
    convergence_accuracy_threshold = 1,
    to_plot = nothing
)

    """
    This experiment runs pmgd training of CIFAR10 with the crossentropy loss function.

    """
    #### Parameter setup
    Random.seed!(seed)
    # Initialize the network architecture, and some random values.
    f_eval_cpu = Chain(
        Conv((3,3), 3=>16, relu; init = Flux.glorot_uniform),
        MaxPool((2,2)),
        Conv((3,3), 16=>32, relu; init = Flux.glorot_uniform),
        MaxPool((2,2)),
        Conv((3,3), 32=>64, relu; init = Flux.glorot_uniform),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(256, 10; init = Flux.glorot_uniform),
        # softmax
    )


    # load the starting parameters to the gpu for wmgd
    # This prevents f_eval from hanging on to the previous run's parameters
    f_eval_wmgd = f_eval_cpu |> gpu



    # write the desired starting parameters into the gpu memory if a starting_params input file is provided
    if !isnothing(starting_params)
        load_parameters(starting_params, f_eval_wmgd)
    end

    # Setup f_cost and accuracy functions
    f_ce(y, y_target) = Flux.Losses.logitcrossentropy(y, y_target)
    f_mse(y, y_target) = Flux.Losses.mse(y, y_target) 
    if cost_type == "crossentropy"
        f_cost = f_ce
    elseif cost_type == "mse"
        f_cost = f_mse
    else
        error("Invalid cost_type")
    end

    accuracy(x, y) = mean(onecold(f_eval_wmgd(x), 1:10) .== onecold(y, 1:10))
    batch_size = batch_size

    #### Select dataset
    label_smoothing = label_smoothing
    Xtrain, Ytrain, Xtest, Ytest = get_cifar10_dataset(test_samples = batch_size,
                                                        batch_size = batch_size,
                                                        label_smoothing = label_smoothing)
    # ###################
    # # wmgd 
    # ###################
    
    logger = Dict()

    logger = wmgd_gpu(
        f_eval = f_eval_wmgd,
        f_cost = f_cost,
        f_dtheta! = f_dtheta!,
        accuracy = accuracy,
        logger = logger,
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = Xtest,
        Ytest = Ytest,
        num_steps = num_steps,
        tau_p = tau_p,
        tau_x = tau_x,
        tau_theta = tau_theta,
        eta = eta,
        norm_magnitude = norm_magnitude,
        seed = seed,
        C_ac_limit = C_ac_limit,
        stopaccuracy = convergence_accuracy_threshold,
        );

    d_now = Dates.now()
    d_now = Dates.format(d_now, "yyyymmdd-HHMM")
    params_filepath = joinpath(results_path, d_now*"end_params.jld2")
    save_parameters(f_eval_wmgd, params_filepath)
    
    title1 = "CIFAR10 using WMGD\n"
    title2 = "τ_θ = $tau_theta / τ_x = $tau_x / norm_mag = $norm_magnitude\n"
    title3 = "batch_size = $batch_size / label_smoothing = $label_smoothing\n"

    plot_logger(logger, 
        title=title1*title2*title3, 
        filename = d_now*"_wmgd_plot.png", 
        to_plot = to_plot, 
        path = results_path)
    
    # Setup parameters
    parameters_dict = Dict(
        :batch_size => batch_size,
        :label_smoothing => label_smoothing,
        :num_steps => num_steps,
        :tau_x => tau_x,
        :tau_theta => tau_theta,
        :tau_p => tau_p,
        :eta => eta,
        :norm_magnitude => norm_magnitude,
        :seed => seed,
        :C_ac_limit => C_ac_limit,
        :cost_type => cost_type,
    )

    # Store any analysis results in a results dictionary
    # and append to results database
    results_dict = merge(
        Dict(:test_type => "CIFAR10-wmgd"),
        analyze_solution_convergence_time(logger, convergence_accuracy_threshold),
        analyze_final_accuracy(logger),
        analyze_max_accuracy(logger),
        analyze_wallclock_time(logger),
        accuracy_factor_ten_progression(logger),
        parameters_dict,
    )
    append_to_results_database("results_database.csv", results_dict)

    # Save any parameters of interest to results_path
    save_function_parameters(results_path, "parameters.txt"; parameters_dict...)

    # Save logger
    save_logger(logger, results_path, "logger.jld2")

    CUDA.reclaim()
    GC.gc(true)
    return results_dict
end


    # ##########
    # # CIFAR10 backprop
    # ##########
function experiment_CIFAR10_backprop(;
    results_path,
    seed = 0,
    batch_size = 1000,
    label_smoothing = 0.01,
    starting_params = nothing,
    num_backprop_epochs = 1000,
    eta = 1,
    convergence_accuracy_threshold = Inf)
    
    #### Parameter setup
    Random.seed!(seed)
    # Initialize the network architecture, and some random values.
    f_eval_cpu = Chain(
        Conv((3,3), 3=>16, relu; init = Flux.glorot_uniform),
        MaxPool((2,2)),
        Conv((3,3), 16=>32, relu; init = Flux.glorot_uniform),
        MaxPool((2,2)),
        Conv((3,3), 32=>64, relu; init = Flux.glorot_uniform),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(256, 10; init = Flux.glorot_uniform),
        # softmax
    )

    #### Choose optimizer
    opt = Flux.Optimise.Descent(eta) # η

    #### Setup f_cost and accuracy functions

    f_eval_backprop = f_eval_cpu |> gpu

    if !isnothing(starting_params)
        load_parameters(starting_params, f_eval_backprop)
    end

    f_cost_backprop(x, y) = Flux.Losses.logitcrossentropy(f_eval_backprop(x), y)
    accuracy_backprop(x, y) = mean(onecold(f_eval_backprop(x), 1:10) .== onecold(y, 1:10))

    #### Select dataset
    label_smoothing = label_smoothing
    Xtrain, Ytrain, Xtest, Ytest = get_cifar10_dataset(test_samples = batch_size,
                                                        batch_size = batch_size,
                                                        label_smoothing = label_smoothing)

    backprop_logger = backprop(;
        num_epochs = num_backprop_epochs,
        stopaccuracy = convergence_accuracy_threshold,
        f_eval = f_eval_backprop,
        f_cost = f_cost_backprop,
        accuracy = accuracy_backprop,
        opt,
        Xtrain,
        Ytrain,
        Xtest,
        Ytest  
    )

    # Save plots to results_path 
    title1 = "CIFAR10 using backprop\n"
    title2 = "eta = $eta / batch_size = $batch_size / label_smoothing = $label_smoothing\n"
    plot_logger(backprop_logger; title=title1*title2, filename = "backprop_plot.png", path = results_path)
    save_logger(backprop_logger, results_path, "backprop_logger.jld2")

    convergence_time = backprop_logger["iteration"][end]
    final_accuracy = analyze_final_accuracy(backprop_logger)
    wallclock_time = analyze_wallclock_time(backprop_logger)
    max_accuracy = analyze_max_accuracy(backprop_logger)
    # Store any analysis results in a results dictionary

    results_dict = Dict()
    results_dict[:convergence_time] = convergence_time
    results_dict[:final_accuracy] = final_accuracy
    results_dict[:wallclock_time] = wallclock_time
    results_dict[:max_accuracy] = max_accuracy
    results_dict[:test_type] = "CIFAR10_backprop"

    # Bookkeeping - write parameters to file in results folder for convenienec
    save_function_parameters(results_path, 
        seed = seed,
        batch_size = batch_size,
        eta = eta,
        label_smoothing = label_smoothing,
        num_backprop_epochs = num_backprop_epochs,
        convergence_time = convergence_time,
        final_accuracy = final_accuracy,
        wallclock_time = wallclock_time,
        max_accuracy = max_accuracy,
        )
    CUDA.reclaim()
    GC.gc(true)
    return results_dict
end
