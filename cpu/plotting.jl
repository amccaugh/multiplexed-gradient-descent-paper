using PyPlot

# pygui(true)
# ion()

""" Plots data stored inside the logger. By specifying `var_names`, you can also
plot subsets of the available variables in the logger """
function plot_data(logger; results_path=nothing, var_names = nothing, x_axis = "t", title = "")
    
    # 
    if var_names === nothing
        # Iterates through all the fields of the DataLogger struct and gets their names
        var_names = [String(sym) for sym in fieldnames(DataLogger)]
        # Remove "log_every_n", "t", and "n" so they're not plotted
        filter!(z->z != "log_every_n", var_names)
        filter!(z->z != "n", var_names)
        filter!(z->z != "t", var_names)
        # Only keep fields of DataLogger which are enabled
        var_names = [name for name in var_names if getfield(logger, Symbol(name)).enabled]
    end

    x = getfield(logger, Symbol(x_axis)).data

    # Plot each of var_names
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["axes.formatter.useoffset"] = false
    fig, axs = subplots(length(var_names), 1, sharex=true)
    for (n,name) in enumerate(var_names)
        if length(axs) > 1
            ax = axs[n]
        else
            ax = axs
        end
        logged_variable = getfield(logger, Symbol(name))
        ax[:plot](transpose(x), transpose(logged_variable.data))
        ax[:set_ylabel](name)
        if name in ["C", "C_ac", "theta", "theta_ac"]
            ax[:axhline](y=0, color="black", linestyle="-", alpha = 0.2)
        end
    end
    axs[1][:set_title](title)
    if !isnothing(results_path)
        file_path = joinpath(results_path, "plot.png")
        file_path_svg = joinpath(results_path, "plot.svg")
        PyPlot.savefig(file_path)
        PyPlot.savefig(file_path_svg)
    end
end