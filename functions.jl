using Distances

using LinearAlgebra
using Plots
using Distributions
using MLKernels
using CuArrays
using Random
function get_approx(X,Xtest,kernf; lambda = 0.0)
    n = size(X)[2]
    mat =  kernelmatrix(Val{:col}(),kernf, X, X)
    if lambda > 0.0
        mat = mat + I *lambda
    end
    v =  kernelmatrix(Val{:col}(),kernf, X, Xtest)
    CL = cholesky(CuArray(Symmetric(mat)))
    pbad = CuArray(v)
    res = CL.U\(CL.L\pbad)
    CuArrays.unsafe_free!(pbad)
    return collect(res)
end


function get_kernel_mat(X,Y,kernf)
    n = size(X)[2]
    mat =  kernelmatrix(Val{:col}(),kernf, X, X)
    CL = cholesky(CuArray(Symmetric(mat)))
    res = CL.U\(CL.L\CuArray(Y))
    return collect(res)
end





function theory(X, Xtest, m)

    @views begin

        n = size(X)[2]
        MS = pairwise(Euclidean(),X).^m


        Mcpu = vcat(Symmetric(MS), ones(1,n))
        M = CuArray(Mcpu)
        MR = qr(M).R

        B = MR\(MR'\M')

        # Now the issue is that we want to achieve that B * vns = 0
        vns = ones(1,n+1)
        vns[end] = 0
        vnscu = CuArray(vns)

        δ = B*vnscu'

        NS = CuArray(nullspace(Mcpu'))
        A = NS'
        α =  δ ./ (A * vnscu')
        Bcor = B .- α .* A


        v =  pairwise(Euclidean(),X,Xtest).^m
        vgpu = cat(CuArray(v),ones(1,size(Xtest)[2]);dims=1)
        return collect((B*vgpu) .- ( α .* (A * vgpu)))
    end
end


function mse_bias_variance(Y_noisy, Ygt, KinvKstar)

    Ypreds = [vec(Y_noisy[i]'KinvKstar) for i = 1:length(Y_noisy)]
    bias =  mean((mean(Ypreds) .- Ygt).^2)
    mse =   mean([ mean((Ypreds[i] .- Ygt).^2) for i = 1:length(Y_noisy)] )
    vars =  mse -  bias
    return mse,bias,vars
end


############################
# Spherical distributions
###########################

struct UniformSphere <: ContinuousMultivariateDistribution
    d
    r
end



function Distributions._rand!(rng::AbstractRNG, dist::UniformSphere,x::AbstractArray{T,2}) where T
    for i =1:size(x)[2]
        u = rand(Normal(0,1),dist.d)  # an array of d normally distributed random variables
        dv= norm(u)
        x[:,i] =  dist.r .* u ./dv
    end
    return x
end

Base.length(us::UniformSphere) = us.d
Distributions.sampler(us::UniformSphere) = us

function _rand!(rng::AbstractRNG, s::UniformSphere, x::AbstractVector{T}) where T<:Real
    u = rand(Normal(0,1),dist.d)  # an array of d normally distributed random variables
    dv= norm(u)
    x[:] =  dist.r .* u ./dv
end

############################
# Bi level Ensemble
###########################


struct BiLevelEnsemble <: ContinuousMultivariateDistribution
    n
    p
    q
    r
    diagonal
    function BiLevelEnsemble(n,p,q,r)
        d =  Integer(floor(n^p))
        s = Integer(floor(n^r))
        a = n^(-q)
        v1 = ones(s) .* (d*a/s)
        v2 = ones(d-s) .*((1-a)*d/(d-s))
        return new(n,p,q,r,vcat(v1,v2)./maximum(vcat(v1,v2)))
    end
end


function Distributions._rand!(rng::AbstractRNG, dist::BiLevelEnsemble,x::AbstractArray{T,2}) where T
    sigma0 = std(Uniform(-1,1))
    rdiag = sqrt.(dist.diagonal)
    for i =1:size(x)[2]
        u = rand(Uniform(-1/sigma0,1/sigma0), length(dist.diagonal))  # an array of d normally distributed random variables
        x[:,i] =  rdiag .* u
    end
    return x
end

Base.length(us::BiLevelEnsemble) = length(us.diagonal)
Distributions.sampler(us::BiLevelEnsemble) = us



############################
# Rakhlin Ensemble
###########################


struct RKEns <: ContinuousMultivariateDistribution
    κ
    d
    diagonal
end


function RKEns(κ,d)
    if κ == 0
        return  RKEns(κ, d, vcat([1.0], [(1-((j-1)/d)^(κ))^(1/κ) for j = 2:d]))
    end

    return RKEns(κ, d, [(1-((j-1)/d)^(κ))^(1/κ) for j = 1:d])
end


function RKEns_from_beta(β,d)
    #find kappa
    κ = 0.2
    el = RKEns(κ,d)
    trs = log(sum(el.diagonal))/log(d)
    c = 1
    lr = 100.0
    if β > 0.9
        counter = 1000
        lr = 1000

    else
        lr = 100
        counter = 100
    end
    stopc = 1

    while abs(trs-β)> 0.001
        κ =  max(0.0, κ - (trs-β)*lr)
        el = RKEns(κ,d)
        trs = log(sum(el.diagonal))/log(d)
        c = c+1
        if c%counter == 0
            lr *= 0.1
        end
        stopc = stopc +1
        if stopc > 10000
            break
        end
    end
    return el

end


function Distributions._rand!(rng::AbstractRNG, dist::RKEns,x::AbstractArray{T,2}) where T
    sigma0 = std(Uniform(-1,1))
    rdiag = sqrt.(dist.diagonal)
    for i =1:size(x)[2]
        u = rand(Uniform(-1/sigma0,1/sigma0), length(dist.diagonal))  # an array of d normally distributed random variables
        x[:,i] =  rdiag .* u
    end
    return x
end

Base.length(us::RKEns) = length(us.diagonal)
Distributions.sampler(us::RKEns) = us
