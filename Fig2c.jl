using CSV
include("functions.jl")
using DataFrames, GLM
using XLSX
include("cv.jl")



X = zeros(1,100)
X[1,:] = 0.0:1/99:1
Y = sin.(2π*X[1,:])


trainidcs = randperm(length(Y))[1:100]
testidc = []
for i = 1:length(Y)
    if !(i in trainidcs)
        push!(testidc, i)
    end
end

Xtrain = X[:,trainidcs]
Xtest = X[:,testidc]
Ytrain = Y[trainidcs]
Ytest = Y[testidc]



# ad dim
function add_dim(Xt,Xtest)
    Xtn = vcat(Xt,rand(1,size(Xt)[2]))
    Xtestn = vcat(Xtest, rand(1,size(Xtest)[2]))
    return Xtn,Xtestn
end


function evaluateplot(kernf,X,Y,Xtest,Ytest;lambda=0.0, binary = false)
    try
        v = get_approx(X,Xtest,kernf;lambda=lambda)
        bn = 0.0
        if binary
            bn = sum(sign.(vec(vec(Y)'*v)) .== Ytest)/length(Ytest)
        end
        return vec(vec(Y)'*v)
    catch
        return 10000000.0
    end
end



function run_exp(tunelambda)
    mse = []
    msecv = []
    τ0 = 0.0
    λ0 = 0.0
    tauls = []
    lambdals = []
    Xtestp = ones(50,1000) *0.5
    Xtestp[1,:] = 0.0:1/999:1
    datapred = []
    Xtn, Xtestn = deepcopy(Xtrain),deepcopy(Xtest)
    for i = 1:50
        τ0,λ0,ms = tune_hyper_params(alphaExp(1.0), Xtn,Ytrain;τ0 = τ0, λ0 = λ0, n = 30, tunelambda = tunelambda)
        push!(tauls,τ0)
        push!(lambdals,λ0)
        push!(msecv,ms)
        kernf = alphaExp(1.0)(τ0, size(Xtn)[1])
        push!(mse, evaluatek(kernf,Xtn,Ytrain,Xtestn,Ytest;lambda = λ0))
        push!(datapred, evaluateplot(kernf,Xtn,Ytrain,Xtestp[1:i,:],Ytest;lambda = λ0))
        Xtn,Xtestn = add_dim(Xtn,Xtestn)

    end
    return tauls,lambdals,msecv,mse, datapred
end





res0 = run_exp(false)
save("data_files/fig2c.jld", "var", res0)
