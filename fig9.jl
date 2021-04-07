
using MAT
using NPZ
include("cv.jl")
include("functions.jl")
using JLD
dictf = matread("data_files/CLL_SUB_111.mat")
X = dictf["X"]
Y = dictf["Y"]
idcsdiscard = vec(Y .!= 1)
X = X[idcsdiscard,:]
Y = Y[:][idcsdiscard]

f1 = 1 .+ npzread("data_files/features1_cll_sub.npy")
f2 = 1 .+ npzread("data_files/features2_cll_sub.npy")
f3 = 1 .+ npzread("data_files/features3_cll_sub.npy")

X2 = X[:,f1]
X3= X[:,f2]
X4 = X[:,f3]
print(size(X4))
X4

#normalise:
for i = 1:size(X4)[2]
    X4[:,i] = X4[:,i]/(mean(abs.(X4[:,i] .- mean(X4[:,i]))))
end

Y = Y .- 2.5
Y = Y*2



n = size(X4)[1]

trainidcs = randperm(length(Y))[1:n]
testidc = []
for i = 1:length(Y)
    if !(i in trainidcs)
        push!(testidc, i)
    end
end

Xtrain = X4'[:,trainidcs]
Xtest = X4'[:,testidc]
Ytrain = Y[trainidcs]
Ytest = Y[testidc]



function run_exp(tunelambda, alpha,Ytrain)
    mse = []
    msecv = []
    τ0 = 0.0
    λ0 = 0.0
    tauls = []
    lambdals = []
    cse = []


    msel = []
    taulsl = []
    lambdalsl = []
    csel = []
    idcs = []

    for j = 1:100
        mse0 = 10000
        lambda0 = λ0
        ce0 = -0.0
        tau0 = 0.0
        msecv0 = 1000

        mse0l = 10000
        lambda0l = λ0
        ce0l = -0.0
        tau0l = 0.0
        i0 = 0
        @views for i = 1:100
            if !(i in idcs)
                idcsn = vcat(idcs,i)
                if j == 1
                    niter = 50
                else
                    niter = 10
                end
                τ0n,λ0n,msn,bn = tune_hyper_params(alphaExp(alpha),Matrix(Xtrain[idcsn,:]),Ytrain;τ0 = τ0, λ0 = λ0, n = niter, tunelambda = true)
                τ0nl,λ0nl,msnl,bnl = tune_hyper_params(alphaExp(alpha),Matrix(Xtrain[idcsn,:]),Ytrain;τ0 = τ0, λ0 = 10e-15, n = niter, tunelambda = false)
                kernf = alphaExp(alpha)(τ0n, i)
                #ms,ce = evaluate(kernf,Xtrain[idcsn,:],Ytrain,Xtest[idcsn,:],Ytest;lambda = λ0n,binary = true)
                if msn < msecv0
                    #mse0 = ms
                    lambda0 = λ0n
                    tau0 = τ0n
                    ce0 = bn
                    msecv0 = msn
                    i0 = i
                    mse0l = msnl
                    tau0l = τ0nl
                    lambda0l = λ0nl
                    ce0l = bnl

                end
            end
        end
        println(i0)
        τ0 = tau0
        λ0 = lambda0



        push!(idcs,i0)
        push!(tauls,τ0)
        push!(lambdals,λ0)
        push!(msecv,msecv0)
        println("lambda0:")
        println(λ0)
        println("tau0:")
        println(τ0)
        push!(msel,mse0l)
        push!(cse,ce0)
        push!(taulsl,tau0l)
        push!(lambdalsl,lambda0l)
        push!(csel,ce0l)
        #Xtn,Xtestn = add_dim(Xtn,Xtestn)
        println("Done with "*string(j))
    end
    return tauls,lambdals,msecv,mse,cse, taulsl,lambdalsl,msel,csel
end


begin


    begin
        alphav = 1.0
        res = run_exp(true, 1.0,Ytrain)
        JLD.save("data_files/fig9.jld","res",res)
    end



    begin
        flipidcs = randperm(length(Ytrain))[1:Integer(floor(length(Y)*0.2))]
        Ytrain[flipidcs] = Ytrain[flipidcs] .* -1
        alphav = 1.0
        res = run_exp(true, 1.0, Ytrain)
        JLD.save("data_files/fig9noise.jld","res",res)
    end

end
