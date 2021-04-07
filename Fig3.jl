include("functions.jl")
using JLD





n = 2000
fun = x -> 2*x[1]^3
#fun = x-> x[1].^2

beta = 0.5
α = 1.0
valsmdl1 = []
valsmdl2 = []
dvals = [500,1000,1500,2000]

begin
    valsmdl1 = []
    rv = -10:0.2:10

    p = plot()
    for (ia,α) in enumerate([ 1.0])
        for (ib,d) in enumerate(dvals)
            mses = []
            zero_pred = 0.0
            begin
                Xc = rand(Normal(), d,n)
                Xtestc = rand(Normal(),d,10000)
                Yc = mapslices(fun, Xc; dims = 1)
                Ytestc = mapslices(fun, Xtestc; dims = 1)
                zero_pred =  mean(vec(Ytestc).^2)
                tra = d

                for τ in rv
                    τus = exp(τ) * 1/tra
                    v = get_approx(Xc,Xtestc,MLKernels.GammaExponentialKernel(τus.^(α/2), α/2))
                    push!(mses, mean( (vec(vec(Yc)'*v) .- vec(Ytestc)).^2 ))
                end
                println(size(mses ./ zero_pred))
                push!(valsmdl1, (mses ./ zero_pred))
            end
        end
    end
end



save("data_files/fig3laplace.jld", "var", valsmdl1)








n = 2000
fun = x -> 2*x[1]^3
#fun = x-> x[1].^2

p = plot()
beta = 0.5
α = 1.0
valsmdl1 = []
valsmdl2 = []
dvals = [500,1000,1500,2000]

begin
    valsmdl1 = []
    rv = -10:0.2:10

    p = plot()
    for (ia,α) in enumerate([ 1.9, 2.0])
        for (ib,d) in enumerate(dvals)
            mses = []
            zero_pred = 0.0
            begin
                Xc = rand(Normal(), d,n)
                Xtestc = rand(Normal(),d,10000)
                Yc = mapslices(fun, Xc; dims = 1)
                Ytestc = mapslices(fun, Xtestc; dims = 1)
                zero_pred =  mean(vec(Ytestc).^2)
                tra = d

                for τ in rv
                    τus = exp(τ) * 1/tra
                    v = get_approx(Xc,Xtestc,MLKernels.GammaExponentialKernel(τus.^(α/2), α/2))
                    push!(mses, mean( (vec(vec(Yc)'*v) .- vec(Ytestc)).^2 ))
                end
                println(size(mses ./ zero_pred))
                push!(valsmdl1, (mses ./ zero_pred))
            end
        end
    end
end



save("data_files/fig3Gauss.jld", "var", valsmdl1)





n = 2000
fun = x -> 2*x[1]^3
#fun = x-> x[1].^2

p = plot()
beta = 0.5
α = 1.0
valsmdl1 = []
valsmdl2 = []
dvals = [500,1000,1500,2000]

begin
    valsmdl1 = []
    rv = -10:0.2:10

    p = plot()
    for (ia,α) in enumerate([ 1.0])
        for (ib,d) in enumerate(dvals)
            mses = []
            zero_pred = 0.0
            begin
                Xc = rand(Normal(), d,n)
                Xtestc = rand(Normal(),d,10000)
                Yc = mapslices(fun, Xc; dims = 1)
                Ytestc = mapslices(fun, Xtestc; dims = 1)
                zero_pred =  mean(vec(Ytestc).^2)
                tra = d

                for τ in rv
                    τus = exp(τ) * 1/tra
                    v = get_approx(Xc,Xtestc,MLKernels.ExponentialKernel(τus))
                    push!(mses, mean( (vec(vec(Yc)'*v) .- vec(Ytestc)).^2 ))
                end
                println(size(mses ./ zero_pred))
                push!(valsmdl1, (mses ./ zero_pred))
            end
        end
    end
end


save("data_files/fig3exp.jld", "var", valsmdl1)
