using Statistics
using Dates
using Flux, Flux.Optimise
using Images: channelview
using Images.ImageCore
using Flux: onehotbatch, onecold, Flux.flatten
using Base.Iterators: partition
using Flux: Momentum
using CUDA
using LinearAlgebra: dot, norm
using Random
using ProgressMeter
using PyPlot
using MLDatasets
using JLD2



function get_cifar10_dataset(;test_samples = 1000, batch_size = 1000, label_smoothing = 0)
    # Download the CIFAR10 dataset -  32 X 32 matrices of numbers in 3 channels (R,G,B)
    
    # Get all the images and labels
    labels = CIFAR10.trainlabels()
    labels = onehotbatch([labels[i] for i in 1:50000], 0:9)

    train_data = permutedims(float(CIFAR10.traintensor()), (2,1,3,4))

    training_samples = 50000 - test_samples
    
    # Break up the images into (batches) of 1000
    idx_val = 50000 - test_samples
    Xtrain = [train_data[:,:,:,p] for p in partition(1:training_samples, batch_size)]
    Ytrain = [labels[:,i] for i in partition(1:idx_val, batch_size)]
    Xtest = train_data[:,:,:,50000-test_samples+1:50000]
    Ytest = labels[:, idx_val+1:end]
    if label_smoothing != 0
        Ytrain = Flux.label_smoothing.(Ytrain, label_smoothing)
        Ytest = Flux.label_smoothing(Ytest, label_smoothing)
    end
    return gpu(Xtrain), gpu(Ytrain), gpu(Xtest), gpu(Ytest)
end


function get_fashion_mnist_dataset(;test_samples = 1000, batch_size = 1000, label_smoothing = 0)
    # Download the FashionMNISt dataset
    
    # Get all the images and labels
    labels = FashionMNIST.trainlabels()
    labels = onehotbatch([labels[i] for i in 1:60000], 0:9)

    train_data = Flux.unsqueeze(float(FashionMNIST.traintensor()), 3)

    training_samples = 60000 - test_samples
    
    # Break up the images into (batches) of 1000
    idx_val = 60000 - test_samples
    Xtrain = [train_data[:,:,:,p] for p in partition(1:training_samples, batch_size)]
    Ytrain = [labels[:,i] for i in partition(1:idx_val, batch_size)]
    Xtest = train_data[:,:,:,60000-test_samples+1:60000]
    Ytest = labels[:, idx_val+1:end]
    if label_smoothing != 0
        Ytrain = Flux.label_smoothing.(Ytrain, label_smoothing)
        Ytest = Flux.label_smoothing(Ytest, label_smoothing)
    end
    return gpu(Xtrain), gpu(Ytrain), gpu(Xtest), gpu(Ytest)
end

""" Creates an XOR (n-bit parity) dataset of bit-size n """
function get_n_bit_parity_dataset(;n = 2)
    if n<10
        num_samples = 2^n-1
    else
        num_samples = 100
    end

    x = [digits(i, base=2, pad=n) for i in 0:num_samples]
    x = hcat(x...)
    y = sum(x, dims = 1).% 2
    num_samples = size(x)[2]

    Xtrain = [copy(z) for z in partition(x, n)] |> gpu
    Ytrain = [copy(z) for z in partition(y, 1)] |> gpu
    Xtest = stack_arrays([copy(z) for z in partition(x, n)]) |> gpu
    Ytest = stack_arrays([copy(z) for z in partition(y, 1)]) |> gpu
    return Xtrain, Ytrain, Xtest, Ytest
end


function finite_difference_gradient(x,y_target, theta, f_eval, f_cost, magnitude)
    G_fd = copy.(theta)
    cost_initial = f_cost(f_eval(x), y_target)
    for i in 1:length(theta)
        for j in 1:length(theta[i])
            theta[i][j] += magnitude
            cost_perturb = f_cost(f_eval(x), y)
            theta[i][j] -= magnitude
            G_fd[i][j] = (cost_perturb - cost_initial)/magnitude
        end
    end
    return G_fd
end

function backprop_gradient(x,y_target,theta,f_eval,f_cost)
    G_backprop = copy.(theta)
    # Use Flux's gradient() function to get 
    gs = gradient(() -> f_cost(f_eval(x), y_target), Flux.params(f_eval))
    for i in 1:length(theta)
        G_backprop[i] .= gs[theta[i]]
    end
    return G_backprop
end

function f_dtheta!(theta_ac::CuArray{Float32}, amplitude::Float32, rng::CUDA.CURAND.RNG)
    # Update theta_ac inplace
    Random.rand!(rng,theta_ac)
    theta_ac .= theta_ac .> 0.5
    theta_ac .-= 0.5
    theta_ac .*= 2*amplitude
end


function wmgd_gpu(;
    f_eval,
    f_cost,
    f_dtheta!,
    accuracy,
    num_steps,
    logger,
    tau_p,
    tau_x,
    tau_theta,
    eta,
    norm_magnitude,
    Xtrain,
    Ytrain,
    Xtest,
    Ytest,
    seed,
    C_ac_limit = Inf,
    stopaccuracy = 1,
    )

    #### Intialize variables
    # Extract the parameters from the Flux Chain
    theta = Flux.params(f_eval)
 
    @show num_params = sum(length.(theta))
    @show amplitude = Float32(norm_magnitude/sqrt(num_params))
    # Make a duplicate copy of the Flux Chain parameters to store the perturbation
    theta_ac = deepcopy.(theta).*0
    G = deepcopy.(theta).*0
    G_int = deepcopy.(theta).*0
    G_true = deepcopy.(theta).*0
    # Other variables
    x = Xtrain[1]
    y_target = Ytrain[1]
    is_x_changed = true
    is_theta_updated = true
    tau_theta_next = 0
    tau_p_next = 1
    tau_x_next = 1
    batch_idx = 0
    num_batches = length(Xtrain)
    # Initialization
    C0 = 0
    C_val = 0
    accuracy_val = 0
    angle = 0
    norm_theta = norm(theta)
    norm_Gint = 0
    C_ac = 0
    start_time = time()
    if typeof(eta) <: Real
        eta_array = fill(eta, num_steps)
    elseif typeof(eta) <: Vector && length(eta) == num_steps
        eta_array = eta
    else
        error("Eta must either be a number or a 1D array of length num_steps")
    end

    # Random numbers
    rng = CURAND.default_rng()
    CUDA.seed!(seed)
    @show hash_parameters(theta)

    if isempty(logger)
        logger["C"] = Float64[]
        logger["C_val"] = Float64[]
        logger["epoch"] = Float64[]
        logger["accuracy_val"] = Float64[]
        logger["iteration"] = Int[]
        logger["eta"] = Float64[]
        logger["angle"] = Float64[]
        logger["norm_Gint"]=Float64[]
        logger["norm_theta"]=Float64[]
        logger["C_ac"] = Float64[]
        logger["y_max"] = Float64[]
        logger["theta_max"] = Float64[]
        logger["time"] = Float64[]
        step_start = 1
    else
        step_start = logger["iteration"][end]
    end

    @showprogress for n in step_start:step_start+num_steps - 1
    
    #### WMGD

        # If more than tau_x has elapsed, update input/target logger
        if n >= tau_x_next
            x = Xtrain[batch_idx % num_batches+1]
            y_target = Ytrain[batch_idx % num_batches+1]
            batch_idx += 1
            tau_x_next += tau_x
            is_x_changed = true
        end

        # Update weights if more than tau_theta has elapsed, or doing lowpass-style weight updates
        if (n >= tau_theta_next)
            # Update θ parameters/weights
            for i in 1:length(theta)
                theta[i] .-= eta_array[n].*G_int[i]
            end
            # Reset the integrated gradient
            for i in 1:length(theta)
                G_int[i] .*= 0
            end
            tau_theta_next += tau_theta
            is_theta_updated = true
        end

        # Add perturbation to theta (without using memory)
        if (n >= tau_p_next)
            for i in 1:length(theta)
                f_dtheta!(theta_ac[i], amplitude, rng) # Update the theta_ac perturbation
                theta[i] .+= theta_ac[i]
            end
            tau_p_next += tau_p
        end

        # Compute y output and the cost
        y = f_eval(x)
        C = f_cost(y, y_target)

        # Remove perturbation from theta (without using memory)
        for i in 1:length(theta)
            theta[i] .-= theta_ac[i]
        end

        #### Calculate C_ac
        # Filter the cost by subtracting a constant offset C0
        if is_x_changed || is_theta_updated
            C0 = f_cost(f_eval(x), y_target)
        end

        C_ac = C-C0
        # Limit the range of C_ac
        C_ac = clamp(C_ac, -C_ac_limit, C_ac_limit)

        for i in 1:length(theta)
            # Compute G, the psuedo-gradient
            G[i] .= C_ac.*theta_ac[i]./(norm_magnitude^2)
            # Compute G_int: integrate G either discretely or lowpass-style
            G_int[i] .+= G[i]
        end

        replace!(G_int, NaN => 0)
        replace!(G_int, Inf => 0)

        if n % 100 == 0
            C_val = f_cost(f_eval(Xtest), Ytest)
            accuracy_val = accuracy(Xtest, Ytest)
            angle  = compute_angle(x,y_target,f_cost,f_eval, theta, G_true,G_int)
            norm_theta = norm(theta)
            norm_Gint = norm(G_int)
        end

        if n % 10000 == 0
            println()
            @show C
            @show C_val
            @show accuracy_val
            # Check for NaN and Inf
            if ~all(all.(isfinite, theta)) || ~isfinite(C)
                return logger
            end
        end
        push!(logger["C"], C)
        push!(logger["C_val"], C_val)
        push!(logger["epoch"], n / num_batches) 
        push!(logger["iteration"], n) 
        push!(logger["accuracy_val"], accuracy_val)
        push!(logger["eta"], eta_array[n])
        push!(logger["angle"], angle)
        push!(logger["norm_theta"], norm_theta)
        push!(logger["norm_Gint"], norm_Gint)
        push!(logger["C_ac"], C_ac)
        push!(logger["y_max"], maximum(abs.(y)))
        push!(logger["theta_max"], maximum([maximum(abs.(n)) for n in theta]))
        push!(logger["time"], time()-start_time)

        is_x_changed = false
        is_p_changed = false
        is_theta_updated = false
        if accuracy_val > stopaccuracy
            return logger
        end
    end

    return logger
end

function plot_logger(logger; title = "", filename = nothing, to_plot = nothing, path = "")
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["axes.formatter.useoffset"] = false
    if isnothing(to_plot)
        to_plot = ["accuracy_val","C_val","C","angle","eta","norm_Gint","norm_theta","C_ac", "y_max", "theta_max"]
    end

    to_plot = [name for name in to_plot if haskey(logger, name)]

    fig, axs = subplots(length(to_plot), 1, sharex=true, figsize=[12,12])
    for (n,name) in enumerate(to_plot)
        axs[n][:plot](logger["iteration"], logger[name])
        axs[n][:set_ylabel](name)
    end
    axs[end][:set_xlabel]("timestep")
    axs[1][:set_title](title)
    if ~isnothing(filename)
        PyPlot.savefig(joinpath(path, filename))
    end
end


function backprop(;
    num_epochs = 100,
    stopaccuracy = 1,
    f_eval,
    f_cost,
    accuracy,
    opt,
    Xtrain,
    Ytrain,
    Xtest,
    Ytest,
    )
    
    logger = Dict()
    logger["C"] = Float64[]
    logger["C_val"] = Float64[]
    logger["epoch"] = Float64[]
    logger["accuracy_val"] = Float64[]
    logger["iteration"] = Float64[]
    logger["eta"] = Float64[]
    logger["time"] = Float64[]
    n = 0

    start_time = time()
    @showprogress for epoch = 1:num_epochs
        @show acc = accuracy(Xtest, Ytest)
        C_val = f_cost(Xtest,Ytest)
        if stopaccuracy !== nothing
            if acc > stopaccuracy
                break 
            end
        end
        # Train on the entire dataset
        for i in 1:length(Xtrain)
            gs = gradient(Flux.params(f_eval)) do
                C = f_cost(Xtrain[i], Ytrain[i])
            end

            Flux.update!(opt, Flux.params(f_eval), gs)
            # Log data
            push!(logger["C"], f_cost(Xtest,Ytest))
            push!(logger["epoch"], epoch) 
            push!(logger["iteration"], n) 
            push!(logger["accuracy_val"], acc)
            push!(logger["C_val"], C_val)
            push!(logger["eta"], opt.eta)
            push!(logger["time"], time()-start_time)
            n += 1

        end
    end

    return logger
end