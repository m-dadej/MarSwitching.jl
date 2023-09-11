
function expected_duration(model::MSM, exog_tvtp::Matrix{Float64} = Matrix{Float64}(undef, 0, 0))

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

function filtered_probs(model::MSM;
                        y::Vector{Float64} = Vector{Float64}(undef, 0),
                        exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                        exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                        exog_tvtp::Matrix{Float64} = Matrix{Float64}(undef, 0, 0))                       
    # TO DO:
    # - check if provided y and exogenous are used in the model
    # - check if y, exogenous have the same number of observations

    if isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y) & isempty(exog_tvtp)
        x = model.x
    else
        T = length(y)
        x = model.intercept == "no" ? [y zeros(T)] : [y ones(T)]        
        x = !isempty(exog_switching_vars) ? [x exog_switching_vars] : x
        x = !isempty(exog_vars) ? [x exog_vars] : x
        x = !isempty(exog_tvtp) ? [x exog_tvtp] : x
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

function predict(model::MSM, 
                 instanteous::Bool = false;
                 y::Vector{Float64} = Vector{Float64}(undef, 0),
                 exog_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_switching_vars::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                 exog_tvtp::Matrix{Float64} = Matrix{Float64}(undef, 0, 0))
    
    # TO DO:
    # - check if provided y and exogenous are used in the model
    # - check if y, exogenous have the same number of observations

    if isempty(exog_vars) & isempty(exog_switching_vars) & isempty(y) & isempty(exog_tvtp)
        ξ_t = filtered_probs(model) 
        T = model.T
        n_δ = Int(length(model.δ)/(model.k*(model.k-1)))
        exog_tvtp = model.x[:, end-n_δ+1:end]
        x = model.x[:,2:end-n_δ]
    else
        T = length(y)
        x = model.intercept == "no" ? zeros(T) : ones(T)       
        x = !isempty(exog_switching_vars) ? [x exog_switching_vars] : x
        x = !isempty(exog_vars) ? [x exog_vars] : x

        ξ_t = filtered_probs(model, y = y, 
                                    exog_vars = exog_vars, 
                                    exog_switching_vars = exog_switching_vars,
                                    exog_tvtp = exog_tvtp) 
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



