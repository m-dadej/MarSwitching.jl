
function convert_arg(var::Symbol; kwargs...)
    return typeof(kwargs[var]) <: Vector ? reshape(kwargs[var], size(kwargs[var])[1], 1) : kwargs[var]
end    

function check_args(model::MSM; kwargs...)

    exog_vars = haskey(kwargs, :exog_vars) ? convert_arg(:exog_vars; kwargs...) : Matrix{Float64}(undef, 0, 0)
    exog_switching_vars = haskey(kwargs, :exog_switching_vars) ? convert_arg(:exog_switching_vars; kwargs...) : Matrix{Float64}(undef, 0, 0)
    exog_tvtp = haskey(kwargs, :exog_tvtp) ? convert_arg(:exog_tvtp; kwargs...) : Matrix{Float64}(undef, 0, 0)

    # for the case when data is provided
    if !isempty(kwargs) #isempty(exog_vars) || isempty(exog_switching_vars) || isempty(y) || isempty(exog_tvtp))
        
        n_δ = Int(length(model.δ)/(model.k*(model.k-1)))

        @assert haskey(kwargs, :y) "y variable cannot be empty when data is provided"
        # argument check: Are provided variables of the same size
        y = kwargs[:y]
        model.n_β_ns > 0 && @assert size(exog_vars) == (size(y)[1], model.n_β_ns) "If data is provided, exog_vars should be a matrix with dimensions ($(size(y)[1]), $(model.n_β_ns))"
        model.n_β > 0 && @assert size(exog_switching_vars) == (size(y)[1], model.n_β) "If data is provided, exog_switching_vars should be a matrix with dimensions ($(size(y)[1]), $(model.n_β))"
        n_δ > 0 && @assert size(exog_tvtp) == (size(y)[1], n_δ) "If data is provided, exog_tvtp should be a matrix with dimensions ($(size(y)[1]), $(Int(length(model.δ)/(model.k*(model.k-1)))))"
    end        

end

"""
    expected_duration(model::MSM, exog_tvtp::VecOrMat{AbstractFloat})

For non-TVTP model, returns Vector of expected duration of each state.
For TVTP model, returns a matrix of expected duration of each state at timt t.    

formula: `1 / (1 - P[i,i])` or for TVTP - `1 / (1 - P[i,i, t])`

# Arguments
- `model::MSM`: estimated model.
- `exog_tvtp::VecOrMat{AbstractFloat}`: optional exogenous variables for the tvtp model. If not provided, in-sample data is used.
"""
function expected_duration(model::MSM, exog_tvtp::VecOrMat{V} = Matrix{Float64}(undef, 0, 0)) where V <: AbstractFloat

    exog_tvtp = typeof(exog_tvtp) <: Vector ? reshape(exog_tvtp, size(exog_tvtp)[1], 1) : exog_tvtp

    if isempty(model.P)
        
        n_δ = Int(length(model.δ)/(model.k*(model.k-1)))

        if isempty(exog_tvtp)
            exog_tvtp = model.x[:, end-n_δ+1:end]   
        end
        
        # this oneliner is faster unfortunately
        T = size(exog_tvtp)[1]
        return reduce(hcat, [(1 ./ (1 .- diag(P_tvtp(exog_tvtp[t, :], model.δ, model.k, n_δ)))) for t in 1:T])'
    else
        return 1 ./ (1 .- diag(model.P))
    end
end

"""
See also [`smoothed_probs`](@ref) and [`expected_duration`](@ref).
"""

function filtered_probs(model::MSM; kwargs...) 
                        
    check_args(model; kwargs...)                        
    
    if isempty(kwargs) #isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y) & isempty(exog_tvtp)
        x = model.x
    else
        T = length(kwargs[:y])
        x = model.intercept == "no" ? [kwargs[:y] zeros(T)] : [kwargs[:y] ones(T)]        
        x = haskey(kwargs, :exog_switching_vars) ? [x kwargs[:exog_switching_vars]] : x
        x = haskey(kwargs, :exog_vars) ? [x kwargs[:exog_vars]] : x
        x = haskey(kwargs, :exog_tvtp) ? [x kwargs[:exog_tvtp]] : x
    end

    if !isempty(model.P)
        ξ = loglik(model.raw_params, 
                    x, 
                    model.k, 
                    model.n_β, 
                    model.n_β_ns, 
                    model.intercept,
                    model.switching_var)[2]
    else
        ξ = loglik_tvtp(model.raw_params, 
                        x, 
                        model.k, 
                        model.n_β,
                        model.n_β_ns, 
                        model.intercept, 
                        model.switching_var, 
                        Int(length(model.δ)/(model.k*(model.k-1))))[2]
    end
    

    return ξ
end


function smoothed_probs(model::MSM; kwargs...)
    
    check_args(model; kwargs...)   
    
    P   = model.P
    k   = model.k
    δ   = model.δ
    n_δ = Int(length(δ)/(k*(k-1)))
                        
    if isempty(kwargs)
        ξ = filtered_probs(model)
        exog_tvtp = isempty(P) ? model.x[:, end-n_δ+1:end] : nothing
        T = model.T
    else
        ξ = filtered_probs(model; kwargs...)   
        exog_tvtp = isempty(P) ? kwargs[:exog_tvtp] : nothing 
        T = size(ξ)[1]
    end
    
    ξ_T      = zeros(size(ξ))
    ξ_T[T,:] = ξ[T, :]

    for t in reverse(1:T-1)
        P          = isempty(P) ? P_tvtp(exog_tvtp[t, :],δ, k, n_δ) : P
        ξ_rate     = ξ_T[t+1, :] ./ (P*ξ[t, :])
        ξ_T[t, :] .= P' * ξ_rate .* ξ[t, :]
    end

    return ξ_T
end

function predict(model::MSM, instanteous::Bool = false; kwargs...)

    check_args(model; kwargs...)   

    if isempty(kwargs)
        ξ_t = filtered_probs(model) 
        T = model.T
        n_δ = Int(length(model.δ)/(model.k*(model.k-1)))
        exog_tvtp = model.x[:, end-n_δ+1:end]
        x = model.x[:,2:end-n_δ]
    else
        T = length(kwargs[:y])
        x = model.intercept == "no" ? zeros(T) : ones(T)       
        x = haskey(kwargs, :exog_switching_vars) ? [x kwargs[:exog_switching_vars]] : x
        x = haskey(kwargs, :exog_vars) ? [x kwargs[:exog_vars]] : x

        ξ_t = filtered_probs(model; kwargs...) 
    end

    if instanteous
        ŷ_s = (x*hcat(model.β...))
    else
        if isempty(model.P)
            for t in 1:T
                P = P_tvtp(exog_tvtp[t, :], model.δ, model.k, Int(length(model.δ)/(model.k*(model.k-1))))
                ξ_t[t, :] = P'ξ_t[t, :]
            end
            ξ_t = ξ_t[1:end-1,:]
        else
            ξ_t = (model.P * ξ_t')'[1:end-1,:]
        end
        ŷ_s  = (x*hcat(model.β...))[2:end,:]
    end

    ŷ = sum(ŷ_s .* ξ_t, dims = 2)

    return ŷ, ξ_t
end



