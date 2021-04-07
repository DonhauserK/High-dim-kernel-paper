include("functions.jl")
using MLJ

function crossvalidate(kernf, X, Y; folds =5,lambda = 0.0)
    #divide set

    try
        kf = Integer(floor(size(X)[2]/folds))
        steps = 1:kf:size(X)[2]
        mses = []
        bn = []


        @views for (i,s) in enumerate(steps)
            if i == length(steps)
                rng = -(kf+steps[i]-size(X)[2])+s+1:size(X)[2]
            else
                rng = s:s+kf-1
            end
            Xt = X[:,[!(s in rng) for s in 1:size(X)[2]]]
            Yt = Y[[!(s in rng) for s in 1:size(X)[2]]]
            Xtest = X[:, rng]
            Ytest = Y[rng]
            v = get_approx(Xt,Xtest,kernf;lambda=lambda)
            push!(mses, mean( (vec(vec(Yt)'*v) .- vec(Ytest)).^2 ) / mean(vec(Ytest).^2))


            push!(bn, sum(sign.(vec(vec(Yt)'*v)) .== Ytest)/length(Ytest))
        end
        return mean(mses), mean(bn)
    catch
        return 10000000.0,0.0
    end

end

function evaluatek(kernf,X,Y,Xtest,Ytest;lambda=0.0, binary = false)

    try

        v = get_approx(X,Xtest,kernf;lambda=lambda)
        bn = 0.0
        if binary
            bn = sum(sign.(vec(vec(Y)'*v)) .== Ytest)/length(Ytest)
        end
        return mean( (vec(vec(Y)'*v) .- vec(Ytest)).^2 ), bn
    catch
        return 10000000.0
    end
end



function gnerate_new_hyper_params(τ0,λ0;tunelambda = true,simga = 0.2)
    if rand(Bernoulli(0.9)) ==0
        τn = rand(Normal(0.0, 1.0))
        λn = λ0
        if tunelambda
            λn = exp( rand(Normal(0.0, 1.0)))
        end
    else
        τn = τ0 + rand(Normal(0.0, simga))
        λn = λ0
        if tunelambda
            λn = exp(log(λn) + rand(Normal(0.0, simga)))
        end
    end
    return max(-50,min(τn,15)), min(λn,50)
end
min(20,50)

a,b = gnerate_new_hyper_params(0.2,0.3)

function get_grid(tunelamb, n)
    if tunelamb
        sq = Integer(floor(sqrt(n)))
        λ,τ = [],[]
        intv = -12:12/(sq-1):0
        intlambda = exp.(-10:10/(sq-1):0)

        for i = 1:sq
            for j = 1:sq
                push!(λ, intlambda[j] .+ 10e-12)
                push!(τ, intv[i])
            end
        end
        for j = 1:n-sq^2+1
            push!(λ,0.0)
            push!(τ,0.0)
        end
        return τ,λ
    else
        return -12:18/(n-1):6,zeros(n)
    end
end



function get_grid2(tunelamb, n)
    if tunelamb
        sq = Integer(floor(sqrt(n)))
        λ,τ = [],[]
        τ = -12 *ones(n)
        λ = exp.(-12:12/(n-1):0)
        return τ,λ
    else
        return -6:6/(n-1):0,zeros(n)
    end
end



function tune_hyper_params_ridge(nkernf, X,Y;τ0 = 0.0, λ0 = -10e6, n = 1000, tunelambda = true)
    if tunelambda && λ0 ==0
        λ0 = 1.0
    end
    kernf = nkernf(τ0, size(X)[1])
    mse0,bn0 = crossvalidate(kernf, X,Y;lambda = λ0)
    #τnls,λnls = get_grid2(tunelambda,n)
    #τnls,λnls  = gnerate_new_hyper_params(τ0,λ0;tunelambda = true,simga = 0.2)

    for i in 1:n
        τn, λn = gnerate_new_hyper_params(τ0,λ0;tunelambda = tunelambda)

        kernf2 = nkernf(τn, size(X)[1])
        mse2,bn2 = crossvalidate(kernf2, X,Y;lambda = λn)
        if mse2 < mse0
            λ0 = λn
            mse0 = mse2
            bn0 = bn2
        end

    end
    return τ0,λ0, mse0,bn0
end


function tune_hyper_params(nkernf, X,Y;τ0 = 0.0, λ0 = -10e6, n = 1000, tunelambda = true)
    if tunelambda && λ0 ==0
        λ0 = 1.0
    end
    kernf = nkernf(τ0, size(X)[1])
    mse0,bn0 = crossvalidate(kernf, X,Y;lambda = λ0)
    #τnls,λnls = get_grid2(tunelambda,n)
    #τnls,λnls  = gnerate_new_hyper_params(τ0,λ0;tunelambda = true,simga = 0.2)

    for i in 1:n
        τn, λn = gnerate_new_hyper_params(τ0,λ0;tunelambda = tunelambda)

        kernf2 = nkernf(τn, size(X)[1])
        mse2,bn2 = crossvalidate(kernf2, X,Y;lambda = λn)
        if mse2 < mse0
            τ0 = τn
            λ0 = λn
            mse0 = mse2
            bn0 = bn2
        end

    end
    return τ0,λ0, mse0,bn0
end

function alphaExp(alpha=1.0)
    return (x,y) -> MLKernels.GammaExponentialKernel((exp(x)/y)^(alpha/2), alpha/2)
end
