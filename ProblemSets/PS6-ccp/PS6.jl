
# worked with Sara Asgari

using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM


function wrapper()

#######################################
#Question 1
#######################################

df = CSV.read("C:\\Users\\ASUS\\Desktop\\metrics3\\fall-2022-1\\ProblemSets\\PS5-ddc\\busdata.csv",DataFrame,header=true,missingstring="")

include("C:\\Users\\ASUS\\Desktop\\metrics3\\fall-2022-1\\ProblemSets\\PS5-ddc\\create_grids.jl")

df = @transform(df, :bus_id = 1:size(df,1))
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, :time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

#######################################
#Question 2
#######################################


θ̂_glm= glm(@formula(Y ~ Odometer * Odometer^2 * RouteUsage * RouteUsage^2 * Branded * time * time^2), df_long, Binomial(), LogitLink())

println(θ̂_glm)

#######################################
#Question 3
#######################################


df3 = CSV.read("C:\\Users\\ASUS\\Desktop\\metrics3\\fall-2022-1\\ProblemSets\\PS5-ddc\\busdata.csv",DataFrame,header=true,missingstring="")

Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])

X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])

Z = Vector(df[:,:RouteUsage])

B = Vector(df[:,:Branded])

N = size(Y,1)

T = size(Y,2)

Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])

Zstate = Vector(df[:,:Zst])

zval,zbin,xval,xbin,xtran = create_grids()

D = DataFrame()
                D.Odometer = kron(ones(zbin), xval)
                D.RouteUsage = kron(zval, ones(xbin))
                D.time= zeros(size(xtran,1))
                D.Branded = zeros(size(xtran,1))
                

function values(din,Zstate,Xstate,xtran,zbin,xbin,xval)        
    
    FV=zeros(size(xtran,1),2,T+1)
    for t=2:T
        for b=0:1
            din.time[t]=t
            din.Branded[t]=b
            p0 = predict(θ̂_glm, din)
        
            FV[:, b+1, t] = - .9 .*log.(p0)
         end
    end
    
    FVT1 = zeros(size(df,1), T)
    for i=1:size(df,1)
        row0 = (Zstate[i]-1)*xbin+1
        for t=1:T
        row1  = Xstate[i,t] + (Zstate[i]-1)*xbin
        FVT1[i,t] = (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,df.Branded[i]+1,t+1]
        end
    end
        
    return FVT1'[:]
end

fvt1=values(D,Zstate,Xstate,xtran,zbin,xbin,xval)

df_long = @transform(df_long,fv = fvt1)

theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded),df_long, Binomial(), LogitLink(),offset=df_long.fv)

end
wrapper()
