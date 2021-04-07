using MAT
using XLSX
include("cv.jl")
include("functions.jl")
using JLD


# https://archive.ics.uci.edu/ml/datasets/Residential+Building+Data+Set
xls = XLSX.readxlsx("data_files/Residential-Building-Data-Set(1).xlsx")
data = xls[1]
datam = data["A3:DE374"]

X4 = datam[:,5:end-2]
Y = datam[:,end-1]

size(X4)

#normalise:
for i = 1:size(X4)[2]
    X4[:,i] = X4[:,i]/(mean(abs.(X4[:,i] .- mean(X4[:,i]))))
end

mat = zeros(size(X4)[2],size(X4)[2])
for j = 1:size(X4)[1]
    global mat = mat + X4[j,:] * X4[j,:]'
end
mat = mat ./ size(X4)[1]
eigen(mat)
tr(mat)

Y = Y./mean(abs.(Y))
X4


n = size(X4)[1]
#n = 100

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

    for j = 1:103
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
        @views for i = 1:103
            if !(i in idcs)
                idcsn = vcat(idcs,i)
                if j == 1
                    niter = 10
                else
                    niter = 5
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


Ytrain = Y[trainidcs]

res = run_exp(true, 1.0,Ytrain)
JLD.save("fig10.jld","res",res)


Ytrain = Y[trainidcs]

Ytrain = Ytrain .+ rand(Uniform(-0.5,0.5), length(Ytrain))
res = run_exp(true, 1.0,Ytrain)
JLD.save("fig10noise.jld","res",res)
