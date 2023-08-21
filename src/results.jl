

function get_std_errors(model::MSM)

    H = FiniteDiff.finite_difference_hessian(θ -> loglik(θ,
                                                         model.x, 
                                                         model.k,
                                                         model.n_β,
                                                         model.n_β_ns,
                                                         model.intercept)[1], model.raw_params) # hessian

    return sqrt.(abs.(diag(pinv(-H))))
end

function expected_duration(model::MSM)
    return 1 ./ (1 .- diag(model.P))
end

function state_coeftable(model::MSM, state::Int64; digits::Int64=3)
    
    # function to clean estimates and provide stats for coefficients
    function coef_clean(coef::Float64, std_err::Float64)
        
        coef       = my_round.(coef)
        std_err    = my_round.(std_err)
        z          = my_round.(coef / std_err)
        pr         = 1-cdf(Chi(1), abs(z))
        pr         = pr < 0.1^(digits) ? "< 1e-$digits" : my_round.(pr)

        return coef, std_err, z, pr
    end
    
    # function to round to digits
    my_round(x) = round(x, digits = digits)

    # header
    println("------------------------------")
    println("Summary of regime $state: ")
    println("------------------------------")
    @printf "Coefficient  |  Estimate  |  Std. Error  |  z value  |  Pr(>|z|) \n"
    @printf "-------------------------------------------------------------------\n"

    if model.intercept == "switching"
        V_σ, V_β, _ = vec2param_switch(get_std_errors(model), model.k, model.n_β, model.n_β_ns)
    else
        V_σ, V_β, _ = vec2param_nonswitch(get_std_errors(model), model.k, model.n_β, model.n_β_ns)
    end

    # β statistics
    for i in 1:length(model.β[state])
        estimate, std_err, z, pr = coef_clean(model.β[state][i], V_β[state][i])
        @printf "%0s%11s%13s%15s%12s%12s\n" "β_$(i-1)" "|" "$estimate  |" "$std_err  |" "$z  |" "$pr  "
    end
    
    # σ statistics
    estimate_σ, σ_std_err, σ_z, σ_pr = coef_clean(model.σ[state], V_σ[state])

    @printf "%0s%13s%13s%15s%12s%12s\n" "σ" "|" "$estimate_σ  |" "$σ_std_err  |" "$σ_z  |" "$σ_pr  "
    @printf "-------------------------------------------------------------------\n"
    @printf "Expected regime duration: %0.2f periods\n" expected_duration(model)[state]
    @printf "-------------------------------------------------------------------\n"

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

    loglik   = round.(model.Likelihood, digits = digits)
    n_params = model.k + model.k*(size(model.x)[2]-1) + (model.k-1)*model.k
    aic      = round.(2*n_params - 2*model.Likelihood, digits = digits)
    bic      = round.(log(model.T)*n_params - 2*model.Likelihood, digits = digits)

    println("Markov Switching Model with $(model.k) regimes")
    @printf "=====================================================\n"
    @printf "%0s%13s%15s%20s\n" "# of observations:" "$(model.T)" "Loglikelihood:" "$loglik"
    @printf "%0s%5s%5s%30s\n" "# of estimated parameters:" "$n_params" "AIC" "$aic"
    @printf "%0s%12s%5s%30s\n" "Error distribution:" "Gaussian" "BIC" "$bic"
    @printf "------------------------------------------------------\n"


    for i in 1:model.k
        state_coeftable(model, i, digits = digits)
    end

    transition_mat(model, digits = digits)
end

