include("functions.jl")
include("cv.jl")
using JLD




n = 500
betav = 0.0:0.01:1.0
dvals = vcat([1],5:5:100)
fun = x ->  0.5*sum(x[1:2:10])^2-sum(x[2:2:11])
noiserv = Uniform(-10.0,10.0)


for iter in 1:1

    for setting in [true,false]
        begin
            msesls = []
            varsls = []
            biasls = []
            zero_pred_ls = []

            p = plot()
            plog = plot()
            mses = []
            biases = []
            vars = []
            zero_pred = []

            for α in [1.0]
                mses = []
                biases = []
                vars = []
                zero_pred = []

                Xtestct = rand(Normal(),dvals[end],10000)
                Xct = rand(Normal(), dvals[end],n)


                Yc = vec(mapslices(fun, Xct; dims = 1))

                Ytestc = vec(mapslices(fun, Xtestct; dims = 1))
                Ynoisy = [vec(Yc) .+ rand(noiserv,n) for i = 1:100]
                λ0n = 1.0e-12
                begin
                    @views for d in dvals

                        Xc = Xct[1:d,:]
                        Xtestc = Xtestct[1:d,:]


                        τ0n,λ0n,_,_ = tune_hyper_params_ridge(alphaExp(α),Xc,Ynoisy[1];τ0 = log(1.0/d), λ0 =λ0n, n = 50, tunelambda = setting)

                        v = get_approx(Xc,Xtestc,MLKernels.GammaExponentialKernel(exp(τ0n), α/2); lambda = λ0n)

                        mse,bias,var = mse_bias_variance(Ynoisy, Ytestc,v)
                        push!(zero_pred, mean(vec(Ytestc).^2))
                        push!(mses, mse)
                        push!(vars,var)
                        push!(biases,bias)
                    end
                    GC.gc()
                    push!(msesls,mses)
                    push!(varsls,vars)
                    push!(biasls,biases)
                    push!(zero_pred_ls,zero_pred)

                end
            end
        end

        save("data_files/fig7"*string(setting)*".jld", "mse", msesls, "bias", biasls, "zero_pred", zero_pred, "vars", varsls)
    end
end
