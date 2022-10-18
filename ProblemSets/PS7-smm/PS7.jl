#worked with Sara Asgari

using SMM

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

function wrapper()

#####################################
# Question1: 
#####################################

 url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
 df = CSV.read(HTTP.get(url).body, DataFrame)
 X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
 y = df.married.==1


function ols_gmm(α, X, y)
    g = y .- X*α
    J = g'*I*g
    return J
end
α̂_optim = optimize(a -> logit_gmm(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
println(α̂_optim.minimizer)


bols = inv(X'*X)*X'*y
println(bols)
# very close results 


#####################################
# Question2: 
#####################################

df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(α, X, y)
        
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)

    for j=1:J
        bigY[:,j] = y.==j
    end

    bigα = [reshape(α,K,J-1) zeros(K)]
    
    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*bigα[:,j])
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    loglike = -sum( bigY.*log.(P) )
        
    return loglike
end


α_rand = rand(6*size(X,2))
α_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
α_start = α_true.*rand(size(α_true))
println(size(α_true))

α_hat_optim = optimize(a -> mlogit(a, X, y), α_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
α_hat_mle = α_hat_optim.minimizer
println(α_hat_mle)

# 2.b

function logit_gmm(α, X, y)
        
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigα = [reshape(α,K,J-1) zeros(K)]

    num   = zeros(N,J)                      
    dem   = zeros(N)                        
    for j=1:J
        num[:,j] = exp.(X*bigα[:,j])
        dem .+= num[:,j]
    end
    P = num./repeat(dem,1,J)
        
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

α_hat_optim = optimize(a -> logit_gmm(a, X, y), α_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
α_hat_mle = α_hat_optim.minimizer
println(α_hat_mle)

# 2.c

α_hat_optim = optimize(a -> logit_gmm(a, X, y), α_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
α_hat_mle = α_hat_optim.minimizer
println(α_hat_mle)

# No, It's not. The coeficients are different. 


#####################################
# Question3: 
#####################################

using Distributions

N=size(X,1)

J = length(unique(y))
Xrand = rand(N,J)
beta = rand(J)
epsilon= randn(N,1)
for i in 1:N
    YY[i] = argmax(X[i]*beta[:] + epsilon[i,:])
end


#####################################
# Question5: 
#####################################

function mlogit_smm(θ, X, y, D)
    K = size(X,2)
    J = length(unique(y))
    N = size(y,1)

    bigY = zeros(N,J)
    for j=1:J
            bigY[:,j] = y.==j
    end
   
    β = bigθ[1:end-1]
    σ = bigθ[end]
    if length(β)==1
        β = β[1]
    end 

    gmodel = zeros(N+1,D)
    gdata  = vcat(bigY,var(bigY))
  
    Random.seed!(1234)  
    for d=1:D
        ε = σ*randn(N,J)
        ỹ = P .+ ε
        gmodel[1:end-1,1:J,d] = ỹ
        gmodel[  end  ,1:J,d] = var(ỹ)
    end

    err = vec(gdata .- mean(gmodel; dims=2))

    J = err'*I*err
    return J
end
  

α_hat_smm = optimize(a -> mlogit_smm(θ, X, y, D), α_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(α_hat_smm)

return nothing
end
wrapper()
