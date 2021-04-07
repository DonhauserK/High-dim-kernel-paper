include("functions.jl")

using Polynomials
using JLD
betav = 0.20:0.05:1.0


function get_optimal_poly(X,Y,Xtest,i=1)
    polyf = Polynomials.fit(X[1,:],vec(Y),i)
    return polyf.(Xtest[1,:])
end



for n in [ 2000,4000,6000]
    pushi= [[] for i = 1:10]
    pushj= [[] for i = 1:10]
    fun = x ->2*x[1]^3

    begin

        for (ialpha, α) in enumerate([1.0])
            begin
                mses = [[],[],[]]
                zero_pred = []
                @views for beta in betav
                    d = max(6,Integer(floor(n^(beta))))
                    Xc = rand(Normal(), d,n)
                    Xtestc = rand(Normal(),d,10000)
                    Yc = mapslices(fun, Xc; dims = 1)
                    Ytestc = mapslices(fun, Xtestc; dims = 1)
                    push!(zero_pred, mean(vec(Ytestc).^2))
                    v = get_approx(Xc,Xtestc,MLKernels.GammaExponentialKernel((1/d)^(α/2), α/2))
                    push!(mses[1], mean((vec((vec(Yc)'*v)) .- vec(Ytestc)).^2))

                    push!(mses[2], mean((get_optimal_poly(Xc,Yc,Xtestc) .- vec(Ytestc)).^2))
                    push!(mses[3], mean((get_optimal_poly(Xc,Yc,Xtestc,2) .- vec(Ytestc)).^2))

                end
            end
            #plot!(plog, betav, log10.(mses./zero_pred), label =string(α)*"-Exp.", legend =:best)
            push!(pushi[1], (mses[1]./zero_pred))
            push!(pushi[2], (mses[2]./zero_pred))
            push!(pushi[3], (mses[3]./zero_pred))

        end
    end
    save("data_files/fig2b"*string(n)*".jld", "var", pushi)
end






n =2000

betav = 0.5:0.05:1.0


for d in [20,40, 60,80]
    pushi= [[] for i = 1:10]
    pushj= [[] for i = 1:10]
    fun = x -> 2*x[1]^2

    begin

        for (ialpha, α) in enumerate([1.0])
            begin
                mses = [[],[],[]]
                zero_pred = []
                @views for beta in betav
                    n= max(6,Integer(floor(d^(1/beta))))

                    Xc = rand(Normal(), d,n)
                    Xtestc = rand(Normal(),d,10000)
                    Yc = mapslices(fun, Xc; dims = 1)
                    Ytestc = mapslices(fun, Xtestc; dims = 1)
                    push!(zero_pred, mean(vec(Ytestc).^2))
                    v = get_approx(Xc,Xtestc,MLKernels.GammaExponentialKernel((1/d)^(α/2), α/2))
                    push!(mses[1], mean((vec((vec(Yc)'*v)) .- vec(Ytestc)).^2))

                    push!(mses[2], mean((get_optimal_poly(Xc,Yc,Xtestc) .- vec(Ytestc)).^2))
                    push!(mses[3], mean((get_optimal_poly(Xc,Yc,Xtestc,2) .- vec(Ytestc)).^2))

                end
            end
            #plot!(plog, betav, log10.(mses./zero_pred), label =string(α)*"-Exp.", legend =:best)
            push!(pushi[1], (mses[1]./zero_pred))
            push!(pushi[2], (mses[2]./zero_pred))
            push!(pushi[3], (mses[3]./zero_pred))

        end
    end
    save("data_files/fig6"*string(d)*".jld", "var", pushi)
end
