module kde

import Optim
op = Optim
import Distances
dist = Distances

export kde_factory, best_kde_sigma, best_balloon_kde_sigma

function distance_matrix(
        x1::Array{Float64, 2},
        x2::Array{Float64, 2},
        distance_metric::Function = (x1_row, x2_row) -> sqrt(sum((x1_row - x2_row) .^ 2))
    )
    @assert size(x1)[2] == size(x2)[2]
    n_output_rows = size(x1)[1]
    n_output_cols = size(x2)[1]
    
    output = zeros((n_output_rows, n_output_cols))
    for i in 1:n_output_rows
        for j in 1:n_output_cols
            output[i, j] = distance_metric(x1[i, :], x2[j, :])
        end
    end
    
    return output
end

function distance_matrix(
        x::Array{Float64, 2},
        distance_metric::Function = (x1_row, x2_row) -> sqrt(sum((x1_row - x2_row) .^ 2))
    )::Array{Float64, 2}
    return distance_matrix(x, x, distance_metric)
end

function default_kde_factory_log_kernel(eval_vs_fit_distance)::Float64
    return di.logpdf(di.Normal(0.0, 1.0), eval_vs_fit_distance)
end

function default_kde_factory_kernel_width(x_row::Vector{Float64})::Float64
    return 1.0
end

function kde_factory(
        ;
        x_fit::Array{Float64, 2},
        log_kernel::Function = default_kde_factory_log_kernel,
        kernel_width::Function = default_kde_factory_kernel_width,
        kernel_width_type::String = ["fixed", "eval", "fit"][1]
    )::Function
    @assert kernel_width_type == "eval" || kernel_width_type == "fit" || kernel_width_type == "fixed"
    n_fit_obs = size(x_fit)[1]
    n_dims = size(x_fit)[2]    

    kernel_widths = zeros(n_fit_obs)
    if kernel_width_type == "fixed"
        @assert kernel_width(zeros(n_dims)) == kernel_width(ones(n_dims))
        @assert isa(kernel_width(zeros(n_dims)), Float64)
        kernel_widths .= kernel_width(zeros(n_dims))
    elseif kernel_width_type == "fit"
        kernel_widths = zeros(n_fit_obs)
        for fit_obs_no in 1:n_fit_obs
            kernel_widths[fit_obs_no] = kernel_width(x_fit[fit_obs_no, :])
        end
    end
    
    if kernel_width_type == "eval"        
        function kde_balloon(x_eval::Array{Float64, 2})::Vector{Float64}
            eval_vs_fit_dm = distance_matrix(x_eval, x_fit)
            n_eval_obs = size(x_eval)[2]
            y_eval = zeros(n_eval_obs)  
            ld = zeros(n_eval_obs)
            for eval_obs_no in 1:n_eval_obs
                kernel_width_value = kernel_width(x_eval[eval_obs_no, :])
                scaled_distances = eval_vs_fit_dm[eval_obs_no, :] ./ kernel_widths
                log_kernel_values = log_kernel.(scaled_distances)
                ld[eval_obs_no] = log_sum_exp(log_kernel_values) - (log(kernel_width_value) * n_dims)
            end

            return ld
        end
        return kde_balloon
    else
        function kde_sample_point(x_eval::Array{Float64, 2})::Vector{Float64}
            eval_vs_fit_dm = distance_matrix(x_eval, x_fit)
            n_eval_obs = size(x_eval)[1]
            y_eval = zeros(n_eval_obs)
            ld = zeros(n_eval_obs)
            for eval_obs_no in 1:n_eval_obs
                scaled_distances = eval_vs_fit_dm[eval_obs_no, :] ./ kernel_widths
                log_kernel_values = log_kernel.(scaled_distances)
                ld[eval_obs_no] = log_sum_exp((log.(kernel_widths) .* n_dims) .+ log_kernel_values)
            end

            return ld
        end
        return kde_sample_point
    end
    
end


function best_kde_sigma(
        ;
        x_fit::Array{Float64, 2},
        true_logpdf::Function,
        init_param::Float64 = 1.0,
        kernel_width::Function = (x_row, param) -> param,
        kernel_width_type::String = "fixed"
    )
    @assert init_param > 0.0
    n_fit_obs = size(x_fit)[1]
    
    function opt_fun(param)
        n_cols = size(x_fit)[2]
        kde_logpdf_values = zeros(n_fit_obs)
        for fit_obs_no in 1:n_fit_obs
            fit_obs_no_set = setdiff(1:n_fit_obs, fit_obs_no)
            kde = kde_factory(
                x_fit = x_fit[fit_obs_no_set, :],
                log_kernel = default_kde_factory_log_kernel,
                kernel_width = (x_row) -> kernel_width(x_row, param),
                kernel_width_type = kernel_width_type
            )
            kde_logpdf_values[fit_obs_no] = kde(reshape(x_fit[fit_obs_no, :], (1, n_cols)))[1]
        end
        true_logpdf_values = map(
            fit_obs_no -> true_logpdf(reshape(x_fit[fit_obs_no, :], (1, n_cols))),
            1:n_fit_obs
        )
        kde_logpdf_values .-= log_sum_exp(kde_logpdf_values)
        true_logpdf_values .-= log_sum_exp(true_logpdf_values)
        dist.JSDivergence()(kde_logpdf_values, true_logpdf_values)
    end
        
    opt = op.optimize(
        opt_fun,
        1e-12,
        2.0,
        op.Brent()
    )
    
    return opt
end


function best_balloon_kde_sigma(
        ;
        x_fit::Array{Float64, 2},
        true_logpdf::Function
    )    
    n_fit_obs = size(x_fit)[1]
    
    fit_vs_fit_dm = distance_matrix(x_fit)
    n_cols = size(x_fit)[2] 
    true_logpdf_values = map(
        fit_obs_no -> true_logpdf(x_fit[fit_obs_no, :]),
        1:n_fit_obs
    )
    
    function log_kernel(scaled_distances::Vector{Float64})::Vector{Float64}
        return di.logpdf.(di.Normal(0.0, 1.0), scaled_distances)
    end
    
    function opt_fun(param)
        kde_logpdf_values = zeros(n_fit_obs)
        for fit_obs_no in 1:n_fit_obs
            distances = fit_vs_fit_dm[fit_obs_no, :]
            log_kernel_width = log(param) - true_logpdf_values[fit_obs_no]
            kernel_width = exp(log_kernel_width)
            scaled_distances = distances ./ kernel_width
            kde_logpdf_values[fit_obs_no] = log_sum_exp(log_kernel(scaled_distances)) - 
                log(n_fit_obs) -
                n_cols * log_kernel_width
        end
        scaled_kde_logpdf_values = kde_logpdf_values .- log_sum_exp(kde_logpdf_values)
        scaled_true_logpdf_values = true_logpdf_values .- log_sum_exp(true_logpdf_values)        
        dist.JSDivergence()(scaled_kde_logpdf_values, scaled_true_logpdf_values)        
    end
    
    opt = op.optimize(
        opt_fun,
        1e-12,
        2.0,
        op.Brent()
    )
    
    return opt
end

end