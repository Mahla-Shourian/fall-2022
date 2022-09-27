#group worked with Sara Asgari

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables,Distributions

function PS4()  

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

#######################################
# Question1
#######################################

# from ps3 solutions

function mlogit_with_Z(theta, X, Z, y)
        
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

#startvalues from ps3, Q1

startvals=[0.05570767876416688, 0.08342649976722213, -2.344887681361976, 0.04500076157943125, 0.7365771540890512, -3.153244238810631, 0.09264606406280998, -0.08417701777996893, -4.273280002738097, 0.023903455659102114, 0.7230648923377259, -3.749393470343111, 0.03608733246865346, -0.6437658344513095, -4.2796847340030375, 0.0853109465190059, -1.1714299392376775, -6.678677013966667, 0.086620198654063, -0.7978777029320784, -4.969132023685069, -0.0941942241795243];

td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward);

# run the optimizer
theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))

theta_hat_mle_ad = theta_hat_optim_ad.minimizer

# evaluate the Hessian at the estimates
H  = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println("logit estimates with Z")
println([theta_hat_mle_ad theta_hat_mle_ad_se]) 

#######################################
# Question2
#######################################
# from PS3 solutions: The coefficient gamma represents the change in utility with a 1-unit change in log wages
# In problem 3 gamma was negative, but here it is positive which makes more sense because an increase in log wage should increase the utility

#######################################
# Question3
#######################################
# 3.a

include("lgwt.jl")
d=Normal(0,1)
nodes, weights = lgwt(7,-4,4)
sum(weights.*pdf.(d,nodes))
sum(weights.*nodes.*pdf.(d,nodes))

# 3.b.1
d=Normal(0,2)
nodes, weights = lgwt(7,-10,10);
sum(weights.*pdf.(d,nodes))
sum(weights.*(nodes.^2).*pdf.(d,nodes))

# 3.b.2
d=Normal(0,2)
nodes, weights = lgwt(10,-10,10);
sum(weights.*pdf.(d,nodes))
sum(weights.*(nodes.^2).*pdf.(d,nodes))

# 3.b.3


# 3.c.1

d=Normal(0,2)
D=1000000
x=rand(Uniform(-10, 10 ),D)
((20/D)*sum(x.^2 .*(pdf.(d,x))))
# It's 3.99985, very close to 4

# 3.c.2

((20/D)*sum(x .*(pdf.(d,x))))
# It's 0.0005, very close to 0

# 3.c.3
((20/D)*sum(pdf.(d,x)))
# It's 0.99941, very close to 1

# 3.c.4
d=Normal(0,2)
D=1000000
x=rand(Uniform(-10, 10 ),D)

((20/D)*sum(x.^2 .*(pdf.(d,x))))
# It's 3.99085, less accurate than when D=1000000

((20/D)*sum(x .*(pdf.(d,x))))
# It's -0.0616, less accurate than when D=1000000

((20/D)*sum(pdf.(d,x)))
# It's 1.013589, less accurate than when D=1000000



#######################################
# Question 4
#######################################

function mlogit_with_Z(theta, X, Z, y,G)
        
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end

    d=Normal(0,1)
    nodes, weights = lgwt(G,-4,4);
    sum(weights.*pdf.(d,nodes))
    sum(weights.*(nodes.^2).*pdf.(d,nodes))
    
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum(log (?))
    
    return loglike
end

#######################################
# Question 5
#######################################


function mlogit_with_Z(theta, X, Z, y,G)
        
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end

    d=Normal(0,1)
    D=1000000
    x=rand(Uniform(-4, 4 ),D)
    ((20/D)*sum(x .*(pdf.(d,x))))
    
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum(log (?))
    
    return loglike
end

end

PS4()

