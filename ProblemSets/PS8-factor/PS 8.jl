#Mahla Shourian (worked with Sara Asgari)
using MultivariateStats, DataFrames, CSV, HTTP, Random, LinearAlgebra, Statistics, Optim, DataFramesMeta, GLM

###################################################
#Question 1
###################################################

df = CSV.read("C:\\Users\\ASUS\\Desktop\\metrics3\\fall-2022-1\\ProblemSets\\PS8-factor\\nlsy.csv",DataFrame,header=true,missingstring="") #2438×15 DataFrame.

ols = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
println(ols)
println(coef(ols))

#[2.0077128963960797, -0.1674410646689739, -0.05424900706962712, -0.15504916236349228, 0.005251015181456092, 0.19564851959073448, 0.2991313425261187]

###################################################
#Question 2
###################################################

ASVAB= convert(Matrix,df[:,r"asvab"])  # 2438×6 Matrix{Float64}:
correlation=cor(ASVAB)  #6×6 Matrix{Float64}:
println(correlation)

#= 
6×6 Matrix{Float64}:
1.0       0.477307  0.799272  0.544505  0.709035  0.677215
0.477307  1.0       0.546088  0.554608  0.500526  0.421961
0.799272  0.546088  1.0       0.638262  0.716732  0.672381
0.544505  0.554608  0.638262  1.0       0.516773  0.461933
0.709035  0.500526  0.716732  0.516773  1.0       0.745446
0.677215  0.421961  0.672381  0.461933  0.745446  1.0
=#

###################################################
#Question 3
###################################################

ols2= lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr+asvabAR+asvabCS+asvabMK+asvabNO+asvabPC+asvabWK), df)
println(coef(ols2))

#[2.1143869429154787, -0.0882103718713371, -0.0026599897179747267, -0.15008247544781522, -0.0021058727747388837, 0.11502177078442746, 0.23511399672742517, 0.04487407066285105, 0.011979736629442243, 0.08006075538506378, 0.006487225850417212, -0.02185546393323712, 0.0018673556800294618]

#=
these ASVAB variables are correlated 
so, if we keep them we'll have measurement error, and if we don't, we will have ommited varibale bias.
=#

###################################################
#Question 4
###################################################

# changing ASVAB matrix into a J.N matrix and reruning the code:

ASVABTR= transpose(ASVAB)  #6×2438 transpose(::Matrix{Float64}) with eltype Float64:
M = fit(PCA, ASVABTR; maxoutdim=1) #PCA(indim = 6, outdim = 1, principalratio = 0.6699194878390354)

asvabPCA = MultivariateStats.transform(M, ASVABTR) #1×2438 Matrix{Float64}:
print(asvabPCA)

#-1.17508  -1.15848  -0.623667  0.596908  4.9637  4.26747  2.11788  4.41202  …  0.550739  -2.27228  -1.8984  0.035048  -1.67309  -2.08833  1.27388  -0.796224
 
#reshaping asvabPCA to add it as a covariate to our model:

asvabPCA= vec(asvabPCA')
insert!(df,7, asvabPCA, :asvabPCA) #2438×16 DataFrame

ols_PCA = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df)
println(coef(ols_PCA))

#[2.124071085655668, -0.09387447133411435, -0.0019219541749570046, -0.16033713309466852, -0.0019598801807180397, 0.10781114940088672, 0.24238319038628725, -0.05253990220858302]
 
 
###################################################
#Question 5
###################################################

M = fit(FactorAnalysis, ASVABTR; maxoutdim=1)  #Factor Analysis(indim = 6, outdim = 1)

asvabFA = MultivariateStats.transform(M, ASVABTR) # 1×2438 Matrix{Float64}:
print(asvabFA)

# 0.442111  0.448392  0.21605  -0.306957  -1.89425  -1.89281  -1.17746  -1.84327  …  -0.461952  1.11378  0.926796  0.120578  0.687666  1.16109 -0.735992  0.375196


#reshaping asvabFA to add it as a covariate to our model:

asvabFA= vec(asvabFA')
insert!(df,8, asvabFA, :asvabFA) #2438×17 DataFram

ols_FA = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFA), df)
println(coef(ols_FA))

#[2.1243208953868313, -0.08578912284905209, 0.0005619454879838437, -0.1559262165394529, -0.002471292173233114, 0.10674639301626562, 0.23926322581637166, 0.11471588115464874]


###################################################
#Question 6
###################################################

include("C:\\Users\\ASUS\\Desktop\\metrics3\\fall-2022-1\\ProblemSets\\PS8-factor\\lgwt.jl") #lgwt 

D= Normal(0,1) #Normal{Float64}(μ=0.0, σ=1.0)
Xa= hcat(df.asvabAR, df.asvabCS,df.asvabMK,df.asvabNO,df.asvabPC,df.asvabWK) #2438×6 Matrix{Float64}
Xw= hcat(ones(size(df,1)), df.black, df.hispanic, df.female) #2438×4 Matrix{Float64}
X= hcat(ones(size(df,1)), df.black, df.hispanic, df.female, df.schoolt, df.gradHS, df.grad4yr) #2438×4 Matrix{Float64}
Y= df.logwage #2438-element Vector{Float64}

function likelihoodfunction(theta,Y, Xa,Xw)
    j=size[Y,2]-1
    KA=size[Xa,2]
    KW=size[Xw,2]

    for j= 1:6
        BetaJ=theta[(j-1)*KA+1:j*KA] 
    end

    BetaW=theta[j*KA+1: J*KA+KW]
    sigmaJ=theta[end-(j-1): end-1]
    sigmaW= theta[end]

    likelihood=zeros(size(df,1),j+1)

    for j= 1: 6
        likelihood[:,j]=log.(pdf.(D, Y-Xa*BetaJ))
    end
    
    function normalmle(theta, Y,X)
        Betha=theta[1:end-1]
        Sigma= thta[end]
        for j= 1: 6
            likelihood[:,end]=log.(pdf.(D, Y-X*Beta))
        end
        
        return likelihood
    end
    
    sumlike = -sum(likelihood)
    return sumlike
end  #likelihoodfunction (generic function with 2 methods)


sigma_hat = optimize(sigma -> -sum(likelihood), rand(1,7), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100, show_trace=true))
println(sigma_hat.minimizer)

# [0.9713869405948758 0.03962940020379291 0.7811631343009768 0.18708331246237786 0.2758933030551407 0.3147748068951862 0.25450883675488145]
