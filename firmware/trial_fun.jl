using Pkg
Pkg.activate("D:/UTD/UTDSummer2023/Calibration-of-LoRa-Nodes-using-ML/")
using Dates, DataFrames, CSV, MLJ, Metrics, LaTeXStrings, StatsPlots, Measures, Distributions, 
ShapML,ScikitLearn, Lathe, GLM
gr()

#Load in dataframe
filepath = "D:/UTD/UTDSummer2023/Calibration-of-LoRa-Nodes-using-ML/data/calibrate.csv"
df = DataFrames.DataFrame(CSV.File(filepath))

#include plotting functions from PlotFunctions.jl and Feature Importance function from FeatureImportance.jl
include("PlotFunctions.jl")
include("FeatureImportance.jl")

#include functions from models file
include("models/LinearRegression.jl")
include("models/NeuralNetworkRegression.jl")
include("models/SVRRegression.jl")
include("models/GaussianProcessRegression.jl")
include("models/DecisionTreeRegression.jl")
include("models/RandomForestRegression.jl")


#variables
y_grimm = []
y_Palas = []
x = []


#Cleaning data. Converting all numeric values in dataframe into Float64.
df.dateTime = map(x -> strip(split(x, '+')[1]), df.dateTime)
df.dateTime = DateTime.(df.dateTime, "dd/mm/yyyy HH:MM:SS")
for col in names(df)
    if eltype(df[:, col]) == Int64  
        df[!, col] = float(df[!, col])
    end
end

#Converts all data in the dataframe to Float32 if the model (like Neural Network Regression) requires it
#=
for col in names(df)
    if eltype(df[:, col]) == Float64 || eltype(df[:, col]) == Int64
        df[!, col] = convert.(Float32, df[!, col])
    end
end
=#


#fill arrays with column names from the dataframe
col_name = names(df)
push!(x, "dateTime")
for i in col_name
    if occursin("_grimm", i)
        push!(y_grimm, i)
    end
    if occursin("_loRa", i)
        push!(x, i)
    end
    if occursin("Palas", i)
        push!(y_Palas, i)
    end
end

#Creating Dictionaries, where key is the value being measured, which is assigned to a dataframe with data from the LoRa Nodes

grimm = Dict{String, DataFrame}()
for i in y_grimm
    grimm_cols = push!(x, i)
    grimm[replace(i, "_grimm" => "")]  = df[:, grimm_cols]
    pop!(x)
end


Palas = Dict{String, DataFrame}()
for i in y_Palas
    Palas_cols = push!(x, i)
    Palas[replace(i, "Palas" => "")] = df[:, Palas_cols]
    pop!(x)
end

# Remove non-PM variables from Palas
Palas_delete = ["pressureh", "temperature", "humidity"]
for key in Palas_delete
    global Palas = delete!(Palas, key)
end

df = DataFrames.select(Palas["pm2_5"], Not("dateTime"))
using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(df,.75)


linearRegressor =  GLM.lm(term(:pm2_5Palas) ~ sum(term.(names(train[!, Not(:pm2_5Palas)]))), train)

r2(linearRegressor)

ypredicted_test = GLM.predict(linearRegressor, test)
ypredicted_train = GLM.predict(linearRegressor, train)

# Test Performance DataFrame (compute squared error)
performance_testdf = DataFrame(y_actual = test[!,:pm2_5Palas], y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

# Train Performance DataFrame (compute squared error)
performance_traindf = DataFrame(y_actual = train[!,:pm2_5Palas], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error ;

# MAPE function defination
function mape(performance_df)
    mape = mean(abs.(performance_df.error./performance_df.y_actual))
    return mape
end


# RMSE function defination
function rmse(performance_df)
    rmse = sqrt(mean(performance_df.error.*performance_df.error))
    return rmse
end


# Train  Error
println("Mean train error: ",mean(abs.(performance_traindf.error)), "\n")
println("Mean Absolute Percentage train error: ",mape(performance_traindf), "\n")
println("Root mean square train error: ",rmse(performance_traindf), "\n")
println("Mean square train error: ",mean(performance_traindf.error_sq), "\n")


# Test Error
println("Mean Absolute test error: ",mean(abs.(performance_testdf.error)), "\n")
println("Mean Aboslute Percentage test error: ",mape(performance_testdf), "\n")
println("Root mean square test error: ",rmse(performance_testdf), "\n")
println("Mean square test error: ",mean(performance_testdf.error_sq), "\n")

# Histogram of error to see if it's normally distributed  on train dataset
histogram(performance_traindf.error, bins = 50, title = "Training Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)


# Histogram of error to see if it's normally distributed  on test dataset
histogram(performance_testdf.error, bins = 50, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)

# Scatter plot of actual vs predicted values on train dataset
train_plot = scatter(performance_traindf[!,:y_actual],performance_traindf[!,:y_predicted], title = "Predicted value vs Actual value on Train Data", ylabel = "Predicted value", xlabel = "Actual value",legend = false)

# Scatter plot of actual vs predicted values on test dataset
test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Predicted value vs Actual value on Test Data", ylabel = "Predicted value", xlabel = "Actual value", legend = false)

fm = term(:pm2_5Palas) ~ sum(term.(names(train[!, Not(:pm2_5Palas)])))
# Cross Validation function defination
using MLBase

function cross_val(train,k, fm = term(:pm2_5Palas) ~ sum(term.(names(train[!, Not(:pm2_5Palas)]))))
    a = collect(MLBase.Kfold(size(train)[1], k))
    for i in 1:k
        row = a[i]
        temp_train = train[row,:]
        temp_test = train[setdiff(1:end, row),:]
        linearRegressor = GLM.lm(fm, temp_train)
        performance_testdf = DataFrame(y_actual = temp_test[!,:pm2_5Palas], y_predicted = GLM.predict(linearRegressor, temp_test))
        performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]

        println("Mean error for set $i is ",mean(abs.(performance_testdf.error)))
    end
end

cross_val(train,10)