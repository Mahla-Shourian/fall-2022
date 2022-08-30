#worked with Sara Asgari

using Random, Distributions, JLD2, CSV, DataFrames,LinearAlgebra,Statistics,FreqTables
Random.seed!(1234)
#Q1:

function q1()
#1.a
A=rand(Uniform(-5,10),(10,7))
B=rand(Normal(-2,15),(10,7))
C=[A[1:5,1:5] B[1:5,6:7]]
D=(A.<=0).*A

#1.b
length(A)

#1.c
unique(D)

#1.d
E=reshape(B,70,1)
E=vec(B)

#1.e
F = reshape([A B], 10, 7, 2)

#1.f
F=permutedims(F,(3,1,2))

#1.g
G=kron(B,C)

#1.h
JLD2.@save "matrixpractice.jld2" A B C D E F G

#1.i
JLD2.@save "firstmatrix.jld2" A B C D

#1.j
df=DataFrame(C,:auto)
CSV.write("Cmatrix.csv",df)

#1.k
dfD=DataFrame(D,:auto)
CSV.write("Dmatrix.dat",dfD,delim=' ')

return A,B,C,D
end
A,B,C,D=q1()

#Q2:
function q2(a,b,c)
#2.a
AB = [A[i,j]*B[i,j] for i in 1:10, j in 1:7]
AB2=A.*B

#2.b
Cprime =  [C[i,j] for i in 1:5, j in 1:7 if -5 <= C[i,j] <= 5]
Cprime2= C[-5 .<= C .<= 5]

#2.c
N, K, T = 15169, 6, 5
X = zeros(N,K,T);
x5 = rand(Binomial(20,0.6),N)
x6 = rand(Binomial(20,0.5),N) 
for k in 1:T 
    
    X[:,1,k] = ones(N)

    if rand(1)[1] <= 75*(6-k)/5
        X[:,2,k] = ones(N)
    end
    X[:,3,k] = rand(Normal(15 + k - 1, 5*(k-1)),N)
    
    X[:,4,k] = rand(Normal(pi*(6-k)/3, 1/exp(1)),N)

    # let columns 1, 5 and 6 remain stationary over time.
    X[:,5, k] = x5
    X[:,6, k] = x6

end

#2.d
β = zeros(K,T);
β[1,:] = [1:0.25:2;];
β[2,:] = [log(j) for j in 1:T];
β[3,:] = [-sqrt(j) for j in 1:T];
β[4,:] = [exp(j) - exp(j+1) for j in 1:T];
β[5,:] = [j for j in 1:T];
β[6,:] = [j/3 for j in 1:T];
β 

#2.e
Y = [X[:,:,t] * β[:,t] + rand(Normal(0,0.36),N,1)  for t in 1:T] 
end
q2(A,B,C)

#Q3

function q3()
#3.a
df= CSV.read("C:\\Users\\ASUS\\Desktop\\metrics3\\fall-2022\\ProblemSets\\PS1-julia-intro\\nlsw88.csv",DataFrame,header=true,missingstring="")
JLD2.@save "nlsw88.jld" df

#3.b
describe(df)
#  neve_married mean = 0.104185 
# 2 obs is missing

#3.c
#tabulate(df,:race)
#tabulate command is undefined.

#3.d
intrq = (describe(df,:q75)[:,2] - describe(df,:q25)[:,2])
summarystats = [describe(df,:nmissing, :mean, :median, :std, :min, :max, :nunique) intrq]  
rename(summarystats, :x1 => :itrq)

#3.e
freqtable(df[:,:industry],df[:,:occupation])

#3.f
df_f=df[:,[:industry, :occupation , :wage]]
gdf=groupby(df_f,[:industry, :occupation])
gdf=combine(gdf, valuecols(gdf) .=> mean)
freqtable(gdf,:wage_mean,:industry,:occupation);
return
end
q3()

#Q4
function q4()
#4.a
fmat = load("firstmatrix.jld2")

#4.b-e
function matrixops(a,b)
    
    if size(a)==size(b)    
        dProduct=a.*b
        product=a'*b
        sprod=sum(a+b)
        return dProduct, product,sprod
    else
        println("inputs must have the same size.")
    end
end

#4.d
matrixops(A,B)

#4.f
matrixops(C,D)

#4.g
JLD2.@load "nlsw88.jld"
matrixops(convert(Array,df.ttl_exp),convert(Array,df.wage));
end
q4()