#worked with Sara Asgari


using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

function ps3()

    #Q1:

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation


function mlogit(Beta, X, Z, y)
    K = size(X,2)+1
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigBeta = [reshape(Beta,K,J-1) zeros(K)]

    bigZ = zeros(N,J)
    for j=1:J
        bigZ[:,j] = Z[:,j]-Z[:,J]
    end

    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        XZ=cat(X,bigZ[:,j],dims=2)
        num[:,j] = exp.(XZ*bigBeta[:,j])
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

mlogit_hat_optim = optimize(b-> mlogit(b,X,Z, y), rand(7*(size(X,2)+1)), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
println(mlogit_hat_optim.minimizer)

#Q2: The coefficient can be interpreted as the change in utility with a percent increase in wage, since we are dealing with a linear-log model.

#Q3:
function nested_logit(alpha, X, Z, y, nesting_structure)

    beta = alpha[1:end-3]
    lambda = alpha[end-2:end-1]
    gamma = alpha[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigBeta = [repeat(beta[1:K],1,length(nesting_structure[1])) repeat(beta[K+1:2K],1,length(nesting_structure[2])) zeros(K)]

    T = promote_type(eltype(X),eltype(alpha))
    num   = zeros(T,N,J)
    lidx  = zeros(T,N,J) 
    dem   = zeros(T,N)

    for j=1:J
        if j in nesting_structure[1]
            lidx[:,j] = exp.( (X*bigBeta[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[1] )
        elseif j in nesting_structure[2]
            lidx[:,j] = exp.( (X*bigBeta[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)./lambda[2] )
        else
            lidx[:,j] = exp.(zeros(N))
        end
    end

    for j=1:J
        if j in nesting_structure[1]
            num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[1][:]];dims=2).^(lambda[1]-1)
        elseif j in nesting_structure[2]
            num[:,j] = lidx[:,j].*sum(lidx[:,nesting_structure[2][:]];dims=2).^(lambda[2]-1)
        else
            num[:,j] = lidx[:,j]
        end

        dem .+= num[:,j]
    end

    P = num./repeat(dem,1,J)

    loglike = -sum( bigY.*log.(P) )

    return loglike
end

nesting_structure = [[1 2 3], [4 5 6 7]]
startvals = [2*rand(2*size(X,2)).-1; 1; 1; .1]

td2 = TwiceDifferentiable(alpha -> nested_logit(alpha, X, Z, y, nesting_structure), startvals; autodiff = :forward)

α_hat_nlogit = optimize(td2, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100, show_trace=true, show_every=50))
α_hat_nlogit_ad = α_hat_nlogit.minimizer
println(α_hat_nlogit_ad)

H  = Optim.hessian!(td2, α_hat_nlogit_ad)
α_hat_nlogit_ad_se = sqrt.(diag(inv(H)))

println([α_hat_nlogit_ad α_hat_nlogit_ad_se])

return nothing

end

ps3()