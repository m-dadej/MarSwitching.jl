
function get_std_errors(model::MSM)

    if !isempty(model.P)
        H = FiniteDiff.finite_difference_hessian(θ -> loglik(θ,
                                                         model.x, 
                                                         model.k,
                                                         model.n_β,
                                                         model.n_β_ns,
                                                         model.intercept,
                                                         model.switching_var)[1], model.raw_params) # hessian
    else
        n_δ = Int(length(model.δ)/(model.k*(model.k-1)))
        H = FiniteDiff.finite_difference_hessian(θ -> loglik_tvtp(θ,
                                                            model.x, 
                                                            model.k,
                                                            model.n_β,
                                                            model.n_β_ns,
                                                            model.intercept,
                                                            model.switching_var, n_δ)[1], model.raw_params) # hessian
    end

    return sqrt.(abs.(diag(mp_inverse(-H))))
end

# function to clean estimates and provide stats for coefficients
function coef_clean(coef::Float64, std_err::Float64, digits::Int64=3)
    
    my_round(x) = round(x, digits = digits)

    coef       = my_round.(coef)
    std_err    = my_round.(std_err)
    z          = my_round.(coef / std_err)
    pr         = 1-cdf(Chi(1), abs(z))
    pr         = pr < 0.1^(digits) ? "< 1e-$digits" : my_round.(pr)

    return coef, std_err, z, pr
end



function state_coeftable(model::MSM, state::Int64; digits::Int64=3)
        
    # function to round to digits
    my_round(x) = round(x, digits = digits)

    # header
    println("------------------------------")
    println("Summary of regime $state: ")
    println("------------------------------")
    @printf "Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|) \n"
    @printf "-------------------------------------------------------------------\n"

    if model.intercept == "switching"
        V_σ, V_β = vec2param_switch(get_std_errors(model), model.k, model.n_β, model.n_β_ns, model.switching_var)
    elseif model.intercept == "non-switching"
        V_σ, V_β = vec2param_nonswitch(get_std_errors(model), model.k, model.n_β, model.n_β_ns, model.switching_var)
    elseif model.intercept == "no"
        V_σ, V_β = vec2param_nointercept(get_std_errors(model), model.k, model.n_β, model.n_β_ns, model.switching_var)
    end

    # β statistics
    for i in 1:length(model.β[state])
        estimate, std_err, z, pr = coef_clean(model.β[state][i], V_β[state][i], digits)
        @printf "%0s%11s%13s%15s%12s%12s\n" "β_$(i-1)" "|" "$estimate  |" "$std_err  |" "$z  |" "$pr  "
    end
    
    # σ statistics
    estimate_σ, σ_std_err, σ_z, σ_pr = coef_clean(model.σ[state], V_σ[state])

    exp_duration = isempty(model.P) ? mean(expected_duration(model)[:, state]) : expected_duration(model)[state]
    @printf "%0s%13s%13s%15s%12s%12s\n" "σ" "|" "$estimate_σ  |" "$σ_std_err  |" "$σ_z  |" "$σ_pr  "
    @printf "-------------------------------------------------------------------\n"
    @printf "Expected regime duration: %0.2f periods\n" exp_duration
    @printf "-------------------------------------------------------------------\n"

end

function coeftable_tvtp(model::MSM; digits::Int64=3)

    # function to round to digits
    my_round(x) = round(x, digits = digits)
    k = model.k
    
    n_δ       = Int(length(model.δ)/(k*(k-1)))
    equations = reshape(model.δ, (k*(k-1)), n_δ)
    exog_tvtp = model.x[:, end-n_δ+1:end]  

    tvtp_intercept = all(exog_tvtp[:,1] .== exog_tvtp[1,1])
    trans_index    = hcat([[j, i] for i in 1:k for j in 1:k-1]...)'

    errors =  get_std_errors(model)[end-(n_δ*k*(k-1))+1:end]
    errors =  reshape(errors, (k*(k-1)), n_δ)

    @printf "Time-varying parameters: \n"
    # header
    println("===================================================================")
    for term in 1:size(equations)[2]

        sign = ((term == 1) & tvtp_intercept) ? "intercept" : "slope"
        println("Summary of term $term ($sign) in TVTP equations:")
        @printf "-------------------------------------------------------------------\n"
        @printf "Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|) \n"
        @printf "-------------------------------------------------------------------\n"
        for i in 1:size(equations)[1]
            
            estimate, std_err, z, pr = coef_clean(equations[i, term], errors[i, term], digits)
            @printf "%0s%0s%0s%2s%13s%15s%12s%12s\n" "δ_$(term - tvtp_intercept)" " [$(trans_index[i,2]) -> " "$(trans_index[i,1])]" "|" "$estimate  |" "$std_err  |" "$z  |" "$pr  "
        end
        @printf "-------------------------------------------------------------------\n"
    end
end

function transition_mat(model::MSM; digits::Int64=2)

    @printf "left-stochastic transition matrix: \n"
    @printf "%20s" " | regime 1"
    
    for s in 2:model.k
        @printf "%13s" " | regime $s"
    end

    @printf "\n"
    for _ in 1:model.k+1
        @printf "-------------"
    end

    #@printf "   |"
    @printf "\n"
    #@printf "----------------------------------------------------------\n"
    for s in 1:model.k
        @printf "%5s" " regime $s |"

        for s2 in 1:model.k
            prob = round.(model.P[s, s2]*100, digits=digits)
            @printf "%10s" "$prob%" 
            @printf "%3s" " |"
        end
        @printf "\n"
    end 
end

function summary_mars(model::MSM; digits::Int64=3)

    r2(ŷ, y) = sum((ŷ .- mean(y)).^2) / sum((y .- mean(y)).^2)
    
    loglik    = round.(model.Likelihood, digits = 1)
    n_params  = length(model.raw_params)
    aic       = round.(2*n_params - 2*model.Likelihood, digits = digits)
    bic       = round.(log(model.T)*n_params - 2*model.Likelihood, digits = digits)
    y         = model.x[:,1]
    n_δ       = Int(length(model.δ)/(model.k*(model.k-1)))
    exog_tvtp = n_δ > 0 ? all(model.x[:, end-n_δ] .== model.x[1, end-n_δ]) : 0 
    k         = model.n_β + model.n_β_ns + n_δ - 1 + exog_tvtp
    step_r2   = r2(Mars.predict(model, false)[1], y[1:end-1])
    inst_r2   = r2(Mars.predict(model, true)[1], y)
    step_r2   = round.((step_r2 - k/(model.T - 1)) * ((model.T - 1)/(model.T - k - 1)), digits = 2)
    inst_r2   = round.((inst_r2 - k/(model.T - 1)) * ((model.T - 1)/(model.T - k - 1)), digits = 2)

    println("Markov Switching Model with $(model.k) regimes")
    @printf "=================================================================\n"
    @printf "%0s%13s%5s%30s\n" "# of observations:" "$(model.T)" "AIC:" "$aic"
    @printf "%0s%5s%5s%30s\n" "# of estimated parameters:" "$n_params" "BIC:" "$bic"
    @printf "%0s%12s%19s%16s\n" "Error distribution:" "Gaussian" "Instant. adj. R^2:" "$inst_r2"
    @printf "%0s%17s%21s%14s\n" "Loglikelihood:" "$loglik" "Step-ahead adj. R^2:" "$step_r2"
    @printf "-----------------------------------------------------------------\n"


    for i in 1:model.k
        state_coeftable(model, i, digits = digits)
    end

    isempty(model.δ) ? transition_mat(model, digits = digits) : coeftable_tvtp(model, digits = digits)
end

