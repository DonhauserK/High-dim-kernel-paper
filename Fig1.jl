include("functions.jl")
using JLD


function rkhsnorm(kern,alpha_s, X)
    mat = kernelmatrix(Val{:col}(), kern, X)
    return alpha_s'mat*alpha_s
end



X= rand(10,5000)


vals = []
plots = []
begin
    prkhs1 = plot()
    normsL = []
    normsExp = []
    n = 5000
    errL = []
    errExp = []
    dvals = 10:5:500
    for d in dvals

        kernL = LaplacianKernel(1/sqrt(d))
        kernExp = ExponentiatedKernel(1/d)
        X = rand(d,n)
        Xtest = rand(d,n)


        Y = X[1,:]


        Ytest = Xtest[1,:]
        alphas = get_kernel_mat(X,Y,kernL)
        push!(normsL, rkhsnorm(kernL,alphas,X))
        v = get_approx(X,Xtest,kernL)
        push!(errL, maximum(abs.(vec(vec(Y)'*v) .- Ytest)))


        alphas = get_kernel_mat(X,Y,kernExp)
        push!(normsExp, rkhsnorm(kernExp,alphas,X))
        v = get_approx(X,Xtest,kernExp)
        push!(errExp, maximum(abs.(vec(vec(Y)'*v) .- Ytest )))

    end
    push!(vals, normsL)
    push!(vals, normsExp)

end

save("data_files/fig1.jld", "L", normsL, "Exp", normsExp)
