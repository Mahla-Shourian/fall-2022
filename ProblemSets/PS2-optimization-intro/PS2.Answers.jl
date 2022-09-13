#worked with Sara Asgari

using Optim,DataFrames, CSV, HTTP,GLM,FreqTables

function ps2()

################    
#Question1
###############

f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
	negf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
	startval = rand(1)   # random starting value
	result = optimize(negf, startval, LBFGS())
	println(result)

##############
#Question2
#############

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)

##############
#Question3
#############

function logit(alpha, X, d)
    log_mat=[d[i]*log((1+exp(-(X*alpha)[i]))^-1)+(1-d[i])*log(1-(1+exp(-(X*alpha)[i]))^-1) for i in 1:size(d,1)]
#     p=(1+exp(-X*alpha))^-1
    loglike=-sum(log_mat)

    return loglike
end
beta_loglike = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=10, show_trace=true))
println(beta_loglike.minimizer)

##############
#Question4
#############

df.white = df.race.==1
α̂ = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
	println(α̂)

##############
#Question5
#############

freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mnlogit(alpha, x, d)
    c=[i==j ? 1 : 0 for i in d , j in 1:6]
    log_mat=[log(exp.(x[i,:]'*alpha[:,j])/(1+sum(exp.(x[i,:]'alpha)))) for i in 1:size(x,1) ,j in 1:6]
    loglike=-sum(c.*log_mat)

    return loglike
end

    beta_mnlog = optimize(b -> mnlogit(b, X, y), rand(size(X,2),6), LBFGS(), Optim.Options(g_tol=1e-7, iterations=10, show_trace=true))
    print(beta_mnlog.minimizer)
end


ps2()

