using MLDatasets
using MAT
using Random

##
""" Create a dataset with data in the shape 
(num_elements_per_sample) x (num samples) """
struct Dataset
    x::Matrix{Float64}
    y_target::Matrix{Float64}
    num_samples::Int
end

XORDataset = Dataset(
    # x
    [0  1  0  1
     0  0  1  1],
    # y_target
    [0 1 1 0],
    # num_samples
    4,
)

FigureDataset = Dataset(
    # x
    [0 .5 0.75
     0 .2 0],
    # y_target
    [0 1 .5],
    # num_samples
    3,
)

""" Creates an XOR (n-bit parity) dataset of bit-size n """
function generate_n_bit_parity_dataset(n)
    x = [digits(i, base=2, pad=n) for i in 0:(2^n-1)]
    x = hcat(x...)
    y = sum(x, dims = 1).% 2
    num_samples = size(x)[2]
    dataset = Dataset(x, y, num_samples)
    return dataset
end

""" Creates an XOR (n-bit parity) dataset of bit-size n """
function generate_simple_dataset(in_size, out_size, num_samples, seed)
    rng = MersenneTwister(seed)
    x = rand(rng, Float64, in_size, num_samples)
    y = rand(rng, Float64, out_size, num_samples)
    dataset = Dataset(x, y, num_samples)
    return dataset
end

function generate_MNIST_training_dataset()
    train_x, train_y = MNIST.traindata(Float64)
    num_samples = length(train_y)

    y_target = zeros((10,num_samples))
    for i in 1:num_samples
        y_target[train_y[i]+1, i] = 1.
    end
    x = reshape(train_x, (784, num_samples))

    MNIST_training_data = Dataset(x, y_target, num_samples)
    
    return MNIST_training_data
end

function generate_NIST_dataset()
    filepath = get_base_path()
    file = matopen(filepath*"/helpers-and-scripts/nistDataSet.mat")
    nistImg = read(file, "nistImg");
    num_samples = size(nistImg)[3]
    y = read(file,"nistLabel");
    y_target = zeros((4,num_samples))
    x = reshape(nistImg, (49, num_samples))
    
    for i in 1:num_samples
        y_target[y[i], i] = 1.
    end

    NIST_data = Dataset(x, y_target, num_samples)
    return NIST_data
end

""" Gets new values from sample number `idx` in the dataset data """
function get_training_batch(idx::Int, dataset::Dataset)
    idx = idx % dataset.num_samples + 1
    x = dataset.x[:, idx]
    y_target = dataset.y_target[:, idx]
    return x, y_target
end

function get_training_batch(idx::Int, dataset::Dataset, batch_size::Int)
    idx_array=Vector{Int}(undef,batch_size)
    for i in 1:batch_size
        idx_array[i] = (idx + 1) % dataset.num_samples + 1
        idx += 1 
    end   
    x = dataset.x[:, idx_array]
    y_target = dataset.y_target[:, idx_array]
    return x,y_target
end


""" In-place version of get_training_batch. Takes in existing vectors `x` and
`y_target` and updates them with new values from sample number `idx` in the
dataset data """
function get_training_batch!(x::Vector{Float64},
        y_target::Vector{Float64},
        idx::Int,
        dataset::Dataset)
    idx = idx % dataset.num_samples + 1
    for n in eachindex(x)
        x[n] = dataset.x[n, idx]
    end
    for n in eachindex(y_target)
        y_target[n] = dataset.y_target[n, idx]
    end
end

function get_training_batch!(x::Array{Float64},
                            y_target::Array{Float64},
                            idx_array::Vector{Int},
                            dataset::Dataset)   

    for n in CartesianIndices(x)
        x[n[1], n[2]]= dataset.x[n[1], idx_array[n[2]]]
    end
    for n in CartesianIndices(y_target)
        y_target[n[1], n[2]]= dataset.y_target[n[1], idx_array[n[2]]]
    end

end

# # Test functions
# x = [-1.0, -1.0]
# y_target = [-1.0]
# @btime update_training_batch!(x, y_target, 3, XORDataset)
# print(y_target)

""" Computes the mean cost by averaging the cost of every example in an entire
dataset """
function mean_cost_over_dataset(dataset::Dataset, f_eval, f_eval_config, f_cost)
    sum_cost::Float64 = 0
    for n in 1:dataset.num_samples
        x = dataset.x[:, n]
        y_target = dataset.y_target[:, n]
        y = f_eval(x, f_eval_config)
        sum_cost += f_cost(y, y_target)
    end
    return sum_cost/dataset.num_samples
end

function mean_cost_over_dataset(dataset::Dataset, theta, f_eval, f_eval_config, f_cost)
    sum_cost::Float64 = 0
    for n in 1:dataset.num_samples
        x = dataset.x[:, n]
        y_target = dataset.y_target[:, n]
        y = f_eval(x, theta, f_eval_config)
        sum_cost += f_cost(y, y_target)
    end
    return sum_cost/dataset.num_samples
end

function classification_accuracy_over_dataset(dataset::Dataset, theta, f_eval, f_eval_config)
    sum_accuracy::Float64 = 0
    for n in 1:dataset.num_samples
        x = dataset.x[:, n]
        y_target = dataset.y_target[:, n]
        y = f_eval(x, theta, f_eval_config)
        m, ind_ytarget = findmax(y_target)
        m, ind_y = findmax(y)
        sum_accuracy += (ind_ytarget==ind_y)
    end
    return sum_accuracy/dataset.num_samples
end

function classification_accuracy_over_dataset(dataset::Dataset, f_eval, f_eval_config)
    sum_accuracy::Float64 = 0
    for n in 1:dataset.num_samples
        x = dataset.x[:, n]
        y_target = dataset.y_target[:, n]
        y = f_eval(x, f_eval_config)
        m, ind_ytarget = findmax(y_target)
        m, ind_y = findmax(y)
        sum_accuracy += (ind_ytarget==ind_y)
    end
    return sum_accuracy/dataset.num_samples
end

function threshold_accuracy_over_dataset(dataset::Dataset, theta, f_eval, f_eval_config)
    sum_accuracy::Float64 = 0
    for n in 1:dataset.num_samples
        x = dataset.x[:, n]
        y_target = dataset.y_target[:, n]
        y = f_eval(x, theta, f_eval_config)
        y_target_thresholded = y_target .> 0.5
        y_thresholded = y .> 0.5
        sum_accuracy += (y_target_thresholded==y_thresholded)
    end
    return sum_accuracy/dataset.num_samples
end


function threshold_accuracy_over_dataset(dataset::Dataset, f_eval, f_eval_config)
    sum_accuracy::Float64 = 0
    for n in 1:dataset.num_samples
        x = dataset.x[:, n]
        y_target = dataset.y_target[:, n]
        y = f_eval(x, f_eval_config)
        y_target_thresholded = y_target .> 0.5
        y_thresholded = y .> 0.5
        sum_accuracy += (y_target_thresholded==y_thresholded)
    end
    return sum_accuracy/dataset.num_samples
end


# # Test functions
# x = [1.30400005, 0.94708096]
# theta = [ 0.08890469, -0.09341224,  0.4528472 ,  0.07417558, -0.43737221,0.29524113,  0.,  0.,  0.]
# config = mlp_config_generator([2,2,1])
# @btime mean_cost(XORDataset, theta, f_eval, f_eval_config, f_cost)

