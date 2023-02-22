##
include("../wmgd.jl")
include("../perturb.jl")

using BenchmarkTools
using PyPlot

# Setup data logging
num_steps = 200000
log_every_n = 1
sparsity = 1
tau_p = 1
n_perturb = 1
dt = 1.0
tau_hp = 1.6
norm_magnitude = 0.05
eta = 0.1
seed = 0

# Select which variables to log/plot
var_names_to_log = [
    # "n",
    # "t",
    "theta",
    "theta_ac",
    # "theta_ac_noisy",
    "x",
    "y_target",
    "y",
    "C",
    "C_ac",
    # "G",
    # "G_int",
    # "G_true",
    # "angle",
    "mean_cost",
    "accuracy",
]

# Choose dataset and test dataset to evaluate
dataset = XORDataset
test_dataset = dataset

# Choose cost function
f_cost = MSE

# Choose size of layers to evaluate and make a MLP network to use as f_eval
layer_sizes = [2,2,1]
f_eval_config = mlp_config_generator(layer_sizes)
f_eval = f_mlp

# Select perturbation function and make a perturbation config struct
f_dtheta! = dtheta_cdma!
f_dtheta_config = dtheta_cdma_config_generator(
    num_params = f_eval_config.theta_length,
    norm_magnitude = norm_magnitude,
    seed = seed,
    n_perturb=tau_p,
    sparsity = 1,
    )

is_analog = false

# Time constants
tau_theta = 1.
tau_x = 1.

# Initialize logger
logger = setup_logger(
        x_length = f_eval_config.x_length, y_length = f_eval_config.y_length,
        theta_length = f_eval_config.theta_length, num_steps = num_steps,
        log_every_n = log_every_n,
        var_names_to_log = var_names_to_log,
        )

# Initialize RNG
rng = MersenneTwister(seed)

# Initialize weights/parameters
theta0 = rand(rng, Float64, f_eval_config.theta_length)

# Execute training
wmgd(f_eval = f_eval,
    f_eval_config = f_eval_config,
    f_cost = f_cost,
    f_dtheta! = f_dtheta!,
    f_dtheta_config = f_dtheta_config,
    logger = logger, 
    dataset = dataset, 
    test_dataset = dataset,
    theta = theta0,
    num_steps = num_steps,
    dt = dt,
    tau_x = tau_x,
    tau_hp = tau_hp,
    tau_theta = tau_theta,
    eta = eta,
    analog = is_analog,
    seed = 0,
    )

# Plot data
plot_data(logger)
