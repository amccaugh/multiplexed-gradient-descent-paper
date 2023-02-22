using LinearAlgebra: dot, norm
using Flux
using TOML
using JLD2
using DataFrames: DataFrame, insertcols!
import LibGit2
import CSV

########################################
### Experiment utilities
########################################



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

""" Locates the parent directory named `name` (e.g. "wmgd").  For instance,
if running a script from /home/myuser/wmgd/tests/test1/, if name is "wmgd"
it will return /home/myuser/wmgd/ """
function get_base_path(name = "wmgd")

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

function create_results_folder(name = "vary-unnamed")
    wmgd_path = get_base_path("wmgd")
    results_path = joinpath(wmgd_path, "results", name)

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


# From https://stackoverflow.com/questions/53120502/julia-create-function-from-string
function function_from_string(s)
    """ Converts any string expression into a function of f(x), such "sin(2.5*x)" """
    f = eval(Meta.parse("x -> " * s))
    return x -> Base.invokelatest(f, x)
end

function generate_eta(;eta_str = "x.+1", num_steps = 100, kwargs...)
    """ Takes a function eta as a string, num_steps and the key,value pairs that
    are the variables in the eta function. Outputs the value of eta"""

    for (key,value) in kwargs
        @eval $key = $value
    end
    x = collect(range(0, 1, num_steps))
    f = function_from_string(eta_str)
    eta = f(x)
    return eta
end


function _angle_between(a, b)
    y::Float32 = dot(a,b)/(norm(a)*norm(b))
    return rad2deg(acos(clamp(y, -1.0, 1.0)))
end


function compute_angle(x,y_target,f_cost,f_eval, theta, G_true,G_int)
    # Use Flux's gradient() function to get 
    gs = gradient(() -> f_cost(f_eval(x), y_target), Flux.params(f_eval))
    for i in 1:length(theta)
        G_true[i] .= gs[theta[i]]
    end

    return _angle_between(G_true,G_int)
end

"""
Takes a vector (1D) of arrays (ND) V and combines them into a single block array (N+1)D
"""
function stack_arrays(V)
    new_dimensions = tuple(size(V[1])...,length(V))
    output = zeros(eltype(V[1]), new_dimensions)
    last_dimension = length(new_dimensions)
    for n in 1:length(V)
        selectdim(output, last_dimension, n) .= V[n]
    end
    return output
end
    
# Reinitialize the weights in a flux network, in this case to glorot uniform and biases to zero.
function reinitialize_params(f_eval)
    for layer in f_eval.layers
        if hasproperty(layer, :weight)
            copyto!(layer.weight, Flux.glorot_uniform(size(layer.weight)))
        end
        if hasproperty(layer, :bias)
            copyto!(layer.bias, zeros(size(layer.bias)))
        end
    end
end

########################################
### Other
########################################




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
function parameter_combinations(;parameters_named_tuple...)
    parameter_dict_list = []
    # If any strings present, don't split them up into individual characters
    parameters_dict = Dict(pairs(parameters_named_tuple))
    for (k,v) in parameters_dict
        if v isa String
            parameters_dict[k] = [v]
        end
    end
    # Create cartesian product of all parameter sweeps
    parameter_values = vec(collect(Base.Iterators.product(values(parameters_dict)...)))
    for pv = parameter_values
        pd = Dict(zip(keys(parameters_dict), pv))
        push!(parameter_dict_list, pd)
    end
    # Convert to dataframe and return
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

""" Takes an arbitrary list of function arguments and saves it to a
TOML-based parameters.txt file at the `path` location.  Called like 
save_function_parameters("./"; a=5, b=2, c="hello") """
function save_function_parameters(path, filename = "parameters.txt"; kwargs...)
    file_path = joinpath(path, filename)
    open(file_path, "a") do io
        TOML.print(io, kwargs)
    end
end

function save_dictionary_parameters(path, dict = Dict{Any,Any}, filename = "parameters.txt")
    file_path = joinpath(path, filename)
    open(file_path, "a") do io
        TOML.print(io, dict)
    end
end


function save_logger(logger, path, filename = "logger.jld2")
    # Save as JLD2 with compression
    filepath = joinpath(path,filename)
    jldsave(filepath, true; logger = logger)
    return filepath
end


function load_logger(path, filename = "logger.jld2")
    # Save as JLD2 with compression
    filepath = joinpath(path,filename)
    logger = jldopen(filepath)["logger"]
    return logger
end

""" Hashes values of parameters θ. Useful for checking reproducible RNG """
function hash_parameters(theta)
    return hash(cpu.(theta))
end

""" Saves the parameters inside `f_eval` to a .jld2 file """
function save_parameters(f_eval, filename)
    theta = Flux.params(f_eval) .|> cpu
    time_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    jldsave(filename*".jld2", theta = cpu.(theta))
end

""" Loads the .jld2 `filename` and puts the loaded parameters into `f_eval` """
function load_parameters(filename, f_eval)
    loaded_theta = load(filename, "theta") |> gpu
    theta = Flux.params(f_eval)
    for i in 1:length(theta)
        theta[i] .= loaded_theta[i]
    end
end


#### Compare saving/loading different file formats
# using CSV
# using DataFrames
# CSV.write("test.csv", DataFrame(logger))
# 

# using BSON
# bson("test.bson", logger)

# using JLD2
# jldsave("test.jld2", true; logger = logger)
# # logger = jldopen("test.jld2")["logger"]

# using HDF5
# h5write("test.h5", "logger", DataFrame(logger))

# using XLSX
# XLSX.writetable("test.xlsx", DataFrame(logger))



function append_to_results_database(filename, results_dict)
    """ Appends a results_dict to a CSV database.  Automatically adds information
    like timestamp, git hash, and git branch """

    filepath = joinpath(get_base_path("wmgd"), filename)

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
