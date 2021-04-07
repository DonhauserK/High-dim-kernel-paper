include("functions.jl")

using Polynomials

n = 4000
betav = 0.20:0.05:1.0



function get_optimal_poly(X,Y,Xtest,i=1)
    polyf = Polynomials.fit(X[1,:],vec(Y),i)
    return polyf.(Xtest[1,:])
end




begin
    pushi= [[[],[],[]] for i = 1:10]
    pushj= [[] for i = 1:10]
    begin

        fun = x ->2*x[1]^2

        p = plot()
        plog = plot()
        mses = []
        zero_pred = []


        for (ialpha, α) in enumerate([0.5, 1.0,1.5, 2.0])
            begin
                mses = [[],[],[]]
                zero_pred = []
                @views for beta in betav
                    d = max(6,Integer(floor(n^(beta))))
                    Xc = rand(Normal(), d,n)
                    for i = 1:size(Xc)[2]
                        Xc[:,i] = sqrt(d) .* Xc[:,i] ./ norm(Xc[:,i])
                    end

                    Xtestc = rand(Normal(),d,10000)
                    for i = 1:size(Xtestc)[2]
                        Xtestc[:,i] =sqrt(d) .* Xtestc[:,i] ./ norm(Xtestc[:,i])
                    end

                    Yc = mapslices(fun, Xc; dims = 1)
                    Ytestc = mapslices(fun, Xtestc; dims = 1)

                    push!(zero_pred, mean(vec(Ytestc).^2))
                    v = get_approx(Xc,Xtestc,MLKernels.GammaExponentialKernel((1/d)^(α/2), α/2))
                    push!(mses[1], mean((vec((vec(Yc)'*v)) .- vec(Ytestc)).^2))

                    push!(mses[2], mean((get_optimal_poly(Xc,Yc,Xtestc) .- vec(Ytestc)).^2))
                    push!(mses[3], mean((get_optimal_poly(Xc,Yc,Xtestc,2) .- vec(Ytestc)).^2))


                end
            end
            push!(pushi[ialpha][1], (mses[1]./zero_pred))
            push!(pushi[ialpha][2], (mses[2]./zero_pred))
            push!(pushi[ialpha][3], (mses[3]./zero_pred))

        end
    end
    using JLD
    save("data_files/fig2a_m3.jld", "var", pushi)



    pushi= [[[],[],[]] for i = 1:10]
    pushj= [[] for i = 1:10]

    begin

        for (ialpha, α) in enumerate([0.5, 1.0,1.5, 2.0])
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
            push!(pushi[ialpha][1], (mses[1]./zero_pred))
            push!(pushi[ialpha][2], (mses[2]./zero_pred))
            push!(pushi[ialpha][3], (mses[3]./zero_pred))

        end
    end
    save("data_files/fig2a_m1.jld", "var", pushi)




    pushi= [[[],[],[]] for i = 1:10]
    pushj= [[] for i = 1:10]


    begin


        for (ialpha,α) in enumerate([0.5, 1.0,1.5, 2.0])
            begin
                mses = [[],[],[]]
                zero_pred = []
                @views for beta in betav
                    ens = RKEns_from_beta(beta,n)
                    Xc = rand(ens, n)
                    Xtestc = rand(ens,10000)
                    Yc = mapslices(fun, Xc; dims = 1)
                    Ytestc = mapslices(fun, Xtestc; dims = 1)
                    push!(zero_pred, mean(vec(Ytestc).^2))
                    v = get_approx(Xc,Xtestc,MLKernels.GammaExponentialKernel(1/(n^(beta))^(α/2), α/2))
                    push!(mses[1], mean((vec((vec(Yc)'*v)) .- vec(Ytestc)).^2))
                    push!(mses[2], mean((get_optimal_poly(Xc,Yc,Xtestc) .- vec(Ytestc)).^2))
                    push!(mses[3], mean((get_optimal_poly(Xc,Yc,Xtestc,2) .- vec(Ytestc)).^2))

                end
            end
            push!(pushi[ialpha][1], (mses[1]./zero_pred))
            push!(pushi[ialpha][2], (mses[2]./zero_pred))
            push!(pushi[ialpha][3], (mses[3]./zero_pred))

        end
    end


    save("data_files/fig2a_m2.jld", "var", pushi)

end
