# Functions in this folder take a logger and return a dictionary of analyzed results

using LinearAlgebra: dot, norm

function analyze_solution_convergence_time(logger, threshold)
    """ Scans through the accuracy data of the logger. Determines the last time the
    accuracy value was below `threshold` """

    accuracy = logger["accuracy_val"]
    idx = findlast(accuracy .< threshold)
    if isnothing(idx) | (idx == length(accuracy))
        t = NaN
    else
        t = logger["iteration"][idx]
    end
    return Dict(:convergence_time => t)
end

function analyze_final_accuracy(logger)
    """ Extracts the final value of accuracy"""

    accuracy = logger["accuracy_val"]
    return Dict(:final_accuracy => accuracy[end])
end

function analyze_max_accuracy(logger)
    """ Extracts the final value of accuracy"""

    accuracy = logger["accuracy_val"]
    return Dict(:max_accuracy => maximum(accuracy))
end

function accuracy_factor_ten_progression(logger)
    """ Extracts the value of accuracy every factor of 10 timesteps"""

    accuracy = logger["accuracy_val"]
    num_steps = length(accuracy)
    num_accuracy_saves = floor(Int,log10(num_steps))
    steps = Vector{Int}(undef,num_accuracy_saves)
    save_accuracy = Vector{Float64}(undef,num_accuracy_saves)

    results_dict = Dict()
    for i in 1:num_accuracy_saves
        # steps[i] = 10^i
        # save_accuracy[i] = accuracy[steps[i]]
        results_dict[Symbol("acc_10^$i")] = accuracy[10^i]
    end
    return results_dict
end

function analyze_wallclock_time(logger)
    """ Extracts the wallclock time for the run"""

    wallclocktime = logger["time"]

    return Dict(:wallclock_time => wallclocktime[end])
end