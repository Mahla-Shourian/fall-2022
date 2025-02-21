# worked with sara asgari

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

include("create_grids.jl") 

# read in function to create state transitions for dynamic model


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# load in the data

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)


# create bus id variable
df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one 
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

theta_hat_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
println(theta_hat_glm)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)


Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

Odo = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]]) 

Xst = Matrix(df[:,[:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, :Xst8, :Xst9, :Xst10, :Xst11, :Xst12, :Xst13, :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]])

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3b: Construct the state transition matrices
#:::::::::::::::::::::::::::::::::::::::::::::::::::

zval,zbin,xval,xbin,xtran = create_grids()

# xtran = (zbin*xbin)* xbin    # Markov transition matrix


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3c: Compute the future value terms
#:::::::::::::::::::::::::::::::::::::::::::::::::::

@views @inbounds function myfun(theta)

N=size(xtran,1)
T=size(Y,2)
FV=zeros(N,2,T+1)
beta=0.9
for t = T:-1:1
    for b = 0:1
        for z = 1:zbin
            for x = 1:xbin
                row=x+(z-1)*xbin
                v1 = theta[1] + theta[2]*xval[x] + theta[3]*b + beta*xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
                v0 = beta*xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
                FV[row,b+1,t]=beta*log(exp(v0)+exp(v1))
            end
        end
    end
end



#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3d: construct the log likelihood
#:::::::::::::::::::::::::::::::::::::::::::::::::::

