##

using BenchmarkTools

struct LoggedVariable
    name::String
    enabled::Bool
    data::Matrix{Float64}
end


struct DataLogger
    n::LoggedVariable
    t::LoggedVariable
    theta::LoggedVariable
    theta_ac::LoggedVariable
    theta_ac_noisy::LoggedVariable
    x::LoggedVariable
    y_target::LoggedVariable
    y::LoggedVariable
    C::LoggedVariable
    C_ac::LoggedVariable
    G::LoggedVariable
    G_int::LoggedVariable
    G_true::LoggedVariable
    angle::LoggedVariable
    angle_full_dataset::LoggedVariable
    mean_cost::LoggedVariable
    accuracy::LoggedVariable
    log_every_n::Int
end

function setup_logged_variable(name, len, num_entries, var_names_to_log)
    enabled = name in var_names_to_log
    if enabled
        data = zeros(Float64, len, num_entries)
    else
        data = zeros(Float64, 0, 0)
    end
    return LoggedVariable(name, enabled, data)
end



function setup_logger(;x_length, y_length, theta_length, num_steps, log_every_n, var_names_to_log)
    num_entries = ceil(Int, num_steps/log_every_n)
    # Guarantee that "t" and "n" are recorded at a minimum
    push!(var_names_to_log, "t")
    push!(var_names_to_log, "n")
    # Create the logger object
    logger = DataLogger(
        setup_logged_variable("n", 1, num_entries, var_names_to_log),
        setup_logged_variable("t", 1, num_entries, var_names_to_log),
        setup_logged_variable("theta", theta_length, num_entries, var_names_to_log),
        setup_logged_variable("theta_ac", theta_length, num_entries, var_names_to_log),
        setup_logged_variable("theta_ac_noisy", theta_length, num_entries, var_names_to_log),
        setup_logged_variable("x", x_length, num_entries, var_names_to_log),
        setup_logged_variable("y_target", y_length, num_entries, var_names_to_log),
        setup_logged_variable("y", y_length, num_entries, var_names_to_log),
        setup_logged_variable("C", 1, num_entries, var_names_to_log),
        setup_logged_variable("C_ac", 1, num_entries, var_names_to_log),
        setup_logged_variable("G", theta_length, num_entries, var_names_to_log),
        setup_logged_variable("G_int", theta_length, num_entries, var_names_to_log),
        setup_logged_variable("G_true", theta_length, num_entries, var_names_to_log),
        setup_logged_variable("angle", 1, num_entries, var_names_to_log),
        setup_logged_variable("angle_full_dataset", 1, num_entries, var_names_to_log),
        setup_logged_variable("mean_cost", 1, num_entries, var_names_to_log),
        setup_logged_variable("accuracy", 1, num_entries, var_names_to_log),
        log_every_n,
    )
    return logger
end


function log_data(
    logger::DataLogger,
    n::Int,
    t::Float64,
    theta::Vector{Float64},
    theta_ac::Vector{Float64},
    theta_ac_noisy::Vector{Float64},
    x::Vector{Float64},
    y_target::Vector{Float64},
    y::Vector{Float64},
    C::Float64,
    C_ac::Float64,
    G::Vector{Float64},
    G_int::Vector{Float64},
    G_true::Vector{Float64},
    angle::Float64,
    angle_full_dataset::Float64,
    mean_cost::Float64,
    accuracy::Float64,
    )

    if ((n-1) % logger.log_every_n) == 0
        idx = floor(Int,(n-1)/logger.log_every_n)+1
        add_entry(logger.n, n, idx)
        add_entry(logger.t, t, idx)
        add_entry(logger.theta, theta, idx)
        add_entry(logger.theta_ac, theta_ac, idx)
        add_entry(logger.theta_ac_noisy, theta_ac_noisy, idx)
        add_entry(logger.x, x, idx)
        add_entry(logger.y_target, y_target, idx)
        add_entry(logger.y, y, idx)
        add_entry(logger.C, C, idx)
        add_entry(logger.C_ac, C_ac, idx)
        add_entry(logger.G, G, idx)
        add_entry(logger.G_int, G_int, idx)
        add_entry(logger.G_true, G_true, idx)
        add_entry(logger.angle, angle, idx)
        add_entry(logger.angle_full_dataset, angle_full_dataset, idx)
        add_entry(logger.mean_cost, mean_cost, idx)
        add_entry(logger.accuracy, accuracy, idx)
    end
end


function add_entry(logged_variable::LoggedVariable, new_data::Union{Vector{Float64},Real}, idx::Int)
    if logged_variable.enabled
        for i in eachindex(new_data)
            logged_variable.data[i,idx] = new_data[i]
        end
    end
end


# function add_entry2(data::Matrix{Float64}, new_data::Vector{Float64}, idx::Int)
#     for i in eachindex(new_data)
#         data[i,idx] = new_data[i]
#     end
# end



# theta =  [1:9.0;]
# theta_ac =  [2:10.0;]
# @btime log_data(
#     logger,
#     1,
#     15, # n
#     0.6, # t
#     theta,
#     theta_ac)

# x = 5

# theta =  [1:9.0;]
# @btime add_entry(logger.theta, theta, 2)

# M = logger.theta.data
# @btime add_entry2(M, theta, 2)
# @btime add_entry2(logger.theta.data, theta, 2)



# @code_lowered M
# @code_lowered logger.theta.data


# ##


# function test(logger)
#     x = 0
#     for n in 1:2
#         x += logger.theta.data[1,1]
#     end
# end
# @btime test(logger)

# function test2(logger)
#     x = 0
#     theta = logger.theta
#     for n in 1:2
#         x += theta.data[1,1]
#     end
# end
# @btime test2(logger)


# function test3(logger)
#     x = 0
#     data = logger.theta.data
#     for n in 1:2
#         x += data[1,1]
#     end
# end
# @btime test3(logger)

