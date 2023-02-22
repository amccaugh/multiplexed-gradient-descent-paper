using LinearAlgebra
using DataFrames
using CSV
using ProgressMeter
using IterTools
using TOML
using JLD2
using LibGit2

include("./logger.jl")

########################################
### Nonlinear activation functions
########################################

sigmoid(x::Float64) = 1/(1+exp(-x))

logistic(x::Float64, x0::Float64, y0::Float64, k::Float64, L::Float64) = L/(1+exp(-k*(x-x0))) + y0

a_relu(x::Float64) = x*(x>0)

########################################
### Cost functions
########################################

# # Vectorized version
# MSE(y::Vector{Float64}, y_target::Vector{Float64}) = sum((y.-y_target).^2)./length(y)

# Non-vectorized version
function MSE(y::Vector{Float64}, y_target::Vector{Float64})
    total::Float64 = 0
    for n in eachindex(y)
        total += (y[n]-y_target[n])^2
    end
    return total/length(y)
end

L2_norm(y::Vector{Float64}, y_target::Vector{Float64}) = norm(y-y_target)

function crossentropy_softmax(y::Vector{Float64}, y_target::Vector{Float64})
    y_softmax  = exp.(y) ./ sum(exp.(y))
    total::Float64 = 0
    for n in eachindex(y_softmax)
        total += -1 * y_target[n] * log(y_softmax[n])
    end
    return total
end

function classification_test(y::Vector{Float64}, y_target::Vector{Float64})
    total::Float64 = 0
    m, ind_target = findmax(y_target)
    m, ind = findmax(y)
    total = ind == ind_target
    return total
end
########################################
### Analyzing logged data
########################################

""" Scans through the mean_cost data of the logger. Determines the last time the
mean_cost was above `threshold`.  num_avg sets how many cost points to average
so that the mean_cost is smoothed/avveraged """

function analyze_final_accuracy_time(logger, threshold)
    accuracy = logger.accuracy.data
    accuracy = dropdims(accuracy, dims = 1)

    t = logger.t.data
    t = dropdims(t, dims = 1)

    idx = findfirst(accuracy .> threshold)
    if isnothing(idx) | (idx == length(accuracy))
        t_sol = NaN
    else
        t_sol = t[idx]
    end
    return Dict(:t_sol => t_sol)
end


""" Extracts the final value of mean_cost """
function analyze_final_cost(logger)
    cost = logger.mean_cost.data
    cost = dropdims(cost, dims = 1)
    return Dict(:final_cost => cost[end])
end

function analyze_final_accuracy(logger)
    accuracy = logger.accuracy.data
    accuracy = dropdims(accuracy, dims = 1)
    return Dict(:final_accuracy => accuracy[end])
end


function accuracy_factor_ten_progression(logger, num_steps)
    """ Extracts the value of accuracy every factor of 10 timesteps"""

    accuracy = logger.accuracy.data
    n = logger.n.data
    accuracy = dropdims(accuracy, dims = 1)
    n = dropdims(n, dims = 1)
    num_accuracy_saves = floor(Int,log10(num_steps))
    steps = Vector{Int}(undef,num_accuracy_saves)
    save_accuracy = Vector{Float64}(undef,num_accuracy_saves)

    results_dict = Dict()
    for i in 1:num_accuracy_saves
        # steps[i] = 10^i
        # save_accuracy[i] = accuracy[steps[i]]
        idx = findlast(n .< 10^i)
        results_dict[Symbol("acc_10^$i")] = accuracy[idx]
    end
    return results_dict
end

function analyze_wallclock_time(logger)
    """ Extracts the wallclock time for the run"""
    wallclock_time = logger.wallclock_time.data
    return Dict(:wallclock_time => wallclock_time[end])
end



function cost_factor_ten_progression(logger, num_steps)
    """ Extracts the value of accuracy every factor of 10 timesteps"""

    accuracy = logger.mean_cost.data
    n = logger.n.data
    accuracy = dropdims(accuracy, dims = 1)
    n = dropdims(n, dims = 1)
    num_accuracy_saves = floor(Int,log10(num_steps))
    steps = Vector{Int}(undef,num_accuracy_saves)
    save_accuracy = Vector{Float64}(undef,num_accuracy_saves)

    results_dict = Dict()
    for i in 1:num_accuracy_saves
        # steps[i] = 10^i
        # save_accuracy[i] = accuracy[steps[i]]
        idx = findlast(n .< 10^i)
        results_dict[Symbol("acc_10^$i")] = accuracy[idx]
    end
    return results_dict
end



########################################
### Parameter conversion / reshaping
########################################


function stack_arrays(V)
    new_dimensions = tuple(size(V[1])...,length(V))
    output = zeros(eltype(V[1]), new_dimensions)
    last_dimension = length(new_dimensions)
    for n in 1:length(V)
        selectdim(output, last_dimension, n) .= V[n]
    end
    return output
end

""" Given an array of layer_sizes, will an array of empty matrices of the
appropriate size for performing weight-matrix multiplication between the layers
"""
function create_weight_matrices(layer_sizes::Vector{Int})
    num_matrices = length(layer_sizes)-1
    matrices = Vector{Matrix{Float64}}(undef, num_matrices)
    for n in 1:num_matrices
        matrices[n] = zeros(layer_sizes[n+1], layer_sizes[n])
    end

    return matrices
end


"""
Given an array of layer_sizes, will an array of empty bias vectors of the
appropriate size for adding biases to each layer
"""
function create_bias_vectors(layer_sizes::Vector{Int})
    num_vectors = length(layer_sizes)-1
    vectors = Vector{Vector{Float64}}(undef, num_vectors)

    for n in 2:length(layer_sizes)
        new_vector = zeros(layer_sizes[n])
        vectors[n-1] = new_vector
    end
    return vectors
end


########################################
### Other
########################################


function finite_difference_gradient(
    f_eval,
    f_eval_config,
    f_cost,
    theta,
    x,
    y_target,
    magnitude,
)

    cost_delta = zeros(Float64, length(theta))

    # Compute initial cost without perturbation)
    y = f_eval(x, theta, f_eval_config)
    cost_initial = f_cost(y, y_target)
    # Iterate through each parameter, perturbing and recomputing costr
    for i in 1:length(theta)
        # Add a perturbation
        theta[i] += magnitude
        # Re-compute cost with new perturbation
        y_perturb = f_eval(x, theta, f_eval_config)
        cost_perturb = f_cost(y_perturb, y_target)
        # Remove perturbation
        theta[i] -= magnitude
        # Compute change in cost
        cost_delta[i] = cost_perturb - cost_initial
    end

    return cost_delta/magnitude
end



# theta = collect(0:8.0)/10
# x = [0, 1.0]
# y_target = [1.0]
# f_eval = f_mlp
# f_eval_config = mlp_config_generator([2,2,1])
# f_cost = MSE
# magnitude = 0.001
# finite_difference_gradient(
#     f_eval,
#     f_eval_config,
#     f_cost,
#     theta,
#     x,
#     y_target,
#     magnitude,
# )


function finite_difference_gradient_full_dataset(
    f_eval,
    f_eval_config,
    f_cost,
    theta,
    dataset,
    magnitude,
)
    grad_sum = zeros(Float64, length(theta))    
    for idx in 1:dataset.num_samples
        x, y_target = get_training_batch(idx, dataset)
        grad = finite_difference_gradient(
            f_eval,
            f_eval_config,
            f_cost,
            theta,
            x,
            y_target,
            magnitude,
        )
        grad_sum .+= grad
    end
    return grad_sum/dataset.num_samples
end


# theta = collect(0:8.0)/10
# dataset = XORDataset
# f_eval = f_mlp
# f_eval_config = mlp_config_generator([2,2,1])
# f_cost = MSE
# magnitude = 0.001
# finite_difference_gradient_full_dataset(
#     f_eval,
#     f_eval_config,
#     f_cost,
#     theta,
#     dataset,
#     magnitude,
# )







function _angle_between(a::Vector{Float64}, b::Vector{Float64})
    # tentative fix for when G_int = 0
    if (norm(a) == 0.) || (norm(b) == 0.) == true
        y = 0.
    else
        y::Float64 = dot(a,b)/(norm(a)*norm(b))
    end

    return rad2deg(acos(clamp(y, -1.0, 1.0)))
end

# clamp.(a,-1.0,1.0)
# a = collect(1:9.0)
# b = collect(2:10.0)
# @btime _angle_between(a,b)




""" Takes a dictionary with lists of parameters and creates combinations
from all the parameters. Useful for generating 1D plots and 2D heatmaps.
The argument num_samples repeats the list for Monte Carlo purposes.

For instance with num_samples = 1:
input:
    parameters_dict = Dict(
        :a => 1,
        :c => [4,5,6],
        :d => [10,30],
    ),
Returns 
    6-element Vector{Any}:
    Dict(:a => 1, :d => 10, :c => 4)
    Dict(:a => 1, :d => 30, :c => 4)
    Dict(:a => 1, :d => 10, :c => 5)
    Dict(:a => 1, :d => 30, :c => 5)
    Dict(:a => 1, :d => 10, :c => 6)
    Dict(:a => 1, :d => 30, :c => 6)
"""
function parameter_combinations(;parameters_dict...)
    parameter_dict_list = []
    parameter_values = vec(collect(Base.Iterators.product(values(parameters_dict)...)))
    for pv = parameter_values
        pd = Dict(zip(keys(parameters_dict), pv))
        push!(parameter_dict_list, pd)
    end
    df = DataFrame(parameter_dict_list)
    return df
end

# parameters_dict = Dict(
#     :a => 1,
#     :c => [4,5,6],
#     :d => [10,30],
# )
# parameter_combinations(parameters_dict)

########################################
### Saving / loading data
########################################

""" Writes a .csv file with the data inside logger """
function write_csv(logger::DataLogger, filename::String)
    # Get all fields of DataLogger
    symbols = [sym for sym in fieldnames(DataLogger)]
    # Remove log_every_n
    filter!(z->z != :log_every_n, symbols)
    # Remove any fields that aren't enabled
    symbols = [sym for sym in symbols if getfield(logger, sym).enabled]
    # Collect all the data matrices
    matrices = [getfield(logger,sym).data for sym in symbols]
    # Create column names for fields
    column_names = String[]
    for (n, sym) in enumerate(symbols)
        name = string(sym)
        num_cols = size(matrices[n], 1)
        if num_cols > 1
            for i in 1:num_cols
                newname = name * string(i)
                push!(column_names, newname)
            end
        else
            push!(column_names, name)
        end
    end
    # Combines all the matrices
    all_data = transpose(vcat(matrices...))

    # Convert to a DataFrame
    df = DataFrame(all_data, column_names)

    # Write data to CSV
    CSV.write(filename, df, delim = ',', header = true)
end


""" Creates a new results folder and returns its path. Will always create a new
    folder based on the largest increment (e.g. if /003/ exists then it
    will create /004/).  For example:

        220419-vary-tau-x/001/
                22-04-19 16-19-32 log.csv
                22-04-19 16-19-32 plot.png
                22-04-19 16-19-32 params.jld2
                22-04-19 16-19-32 parameters.txt
        220419-vary-tau-x/002/
                22-07-22 16-19-32 log.csv
                22-07-22 16-19-32 plot.png
                22-07-22 16-19-32 params.jld2
                22-07-22 16-19-32 parameters.txt
"""
function create_results_subfolder(name = "vary-unnamed")
    results_path = create_results_folder(name)
    i = 1
    while true
        new_folder = lpad(i,3,"0")
        path = joinpath(results_path, new_folder)
        if !ispath(path)
            mkpath(path)
            return path
        end
        i += 1
    end
end

function create_results_folder(name = "vary-unnamed", root = "wmgd-cpu")
    # Get current path and split into individual directory names
    path_components = splitpath(@__DIR__)
    # Find the folder "wmgd" or "wmgd-cpu"
    wmgd_idx = findfirst(s->s==root, path_components)
    # If "wmgd" doesn't exist, throw error
    if isnothing(wmgd_idx)
        error("Error in create_results_folder(): not running from path inside wmgd folder, currently $(@__DIR__)")
    else
        # Build results path
        wmgd_path = joinpath(path_components[1:wmgd_idx])
        results_path = joinpath(wmgd_path, "results", name)
    end

    if !ispath(results_path)
        mkpath(results_path)
    end
    return results_path
end


""" Recursively searches through `path` to find files that have the filename `filename`
and returns paths that contain that filename """
function find_files(path, filename)
    file_paths = []
    for (root, dirs, files) in walkdir(path)
        for file in files
            if file == filename
                push!(file_paths, joinpath(results_path,root))
            end
        end
    end
    return file_paths
end

function append_to_results_database(filename, results_dict)
    """ Appends a results_dict to a CSV database.  Automatically adds information
    like timestamp, git hash, and git branch """

    filepath = joinpath(get_base_path("wmgd-cpu"), filename)

    # Check if database CSV file exists, if not make a blank one
    if isfile(filepath)
        df = DataFrame(CSV.File(filepath))
    else
        df = DataFrame()
    end
    new_df = DataFrame(results_dict)

    # Add time/date columns
    insertcols!(new_df, 1, :time => time())
    insertcols!(new_df, 2, :timestr => Dates.format(Dates.now(), "e U dd, yyyy HH:MM"))
    # Add user
    insertcols!(new_df, 3, :user => splitdir(homedir())[end])
    # Add filename
    insertcols!(new_df, 4, :filename => joinpath(splitpath(@__FILE__)[end-1:end]))
    # Add git commit version
    repo = LibGit2.GitRepo("./")
    head = LibGit2.head(repo)
    githash = string(LibGit2.GitHash(head))
    gitbranch = LibGit2.headname(repo)
    insertcols!(new_df, 5, :commit => githash[1:7])
    insertcols!(new_df, 6, :branch => gitbranch)

    # Join using vertical concatenation (cols=:union means allow new columns)
    combined_df = vcat(df, new_df, cols = :union)
    # Save file to CSV
    CSV.write(filepath, combined_df)
end

    """ Takes an arbitrary list of function arguments and saves it to a
TOML-based parameters.txt file at the `path` location.  Called like 
save_function_parameters("./"; a=5, b=2, c="hello") """
function save_function_parameters(path, filename = "parameters.txt"; kwargs...)
    file_path = joinpath(path, filename)
    open(file_path, "a") do io
        TOML.print(io, kwargs)
    end
end

function get_base_path(name = "wmgd-cpu")

    # Get current path and split into individual directory names
    path_components = splitpath(@__DIR__)
    # Find the folder e.g. "wmgd"
    folder_idx = findfirst(s->s==name, path_components)
    # If "wmgd" doesn't exist, throw error
    if isnothing(folder_idx)
        error("Error in create_results_folder(): not running from path inside '$name' folder, currently '$(@__DIR__)'")
    else
        folder_path = joinpath(path_components[1:folder_idx])
    end

    return folder_path

end

##############
# Make plots consistant
##############

function setup_plot_params()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    font0 = Dict(
            "font.size" => 25,
            "axes.labelweight" => "bold",
            "axes.labelsize" => 20,
            "lines.linewidth" => 2,
            "xtick.labelsize" => 12,
            "ytick.labelsize" => 12,
            "legend.fontsize" => 12,
            "figure.figsize" => [6, 25],
    )
    merge!(rcParams, font0)
end

