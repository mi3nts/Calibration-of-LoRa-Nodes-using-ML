GaussianProcessRegressor = @load GaussianProcessRegressor pkg=MLJScikitLearnInterface verbosity = 0

# Gaussian Process Regression Function

function GaussianProcessRegression(k, X_train, y_train, X_test, y_test, wholedata)
    
    # Training model
    gpr = machine(GaussianProcessRegressor(), X_train, vec(Matrix(y_train)))
    MLJ.fit!(gpr, verbosity = 0)
    predict_train = MLJ.predict(gpr, X_train)
    predict_test = MLJ.predict(gpr, X_test)
    
    #Print r2, mse, and rmse values for test data
    #r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    #println("Gaussian Process Regression: test r2 value for " * k * " = " * string(r2_score_test))
    mse_test = round(mse(predict_test, Matrix(y_test)), digits=3)
    println("Gaussian Process Regression: test mse value for " * k * " = " * string(mse_test))
    rmse_test = sqrt(mse_test)
    println("Gaussian Process Regression: test rmse value for " * k * " = " * string(rmse_test))


    # Calculating Feature Importance using the FeatureImportance Function from FeatureImportance.jl
    data_plot = FeatureImportance(wholedata, k, gpr)


    #LaTex Formatting
    if k[1:2] == "pm" && k != "pmTotal"
        if occursin(k, "pm2_5")
            k = replace(k, "_" => ".")
        end
        k = "pm" * latexstring("_{" * k[3:length(k)] * "}") * " (µg/m" * latexstring("^{3}") * ")"
    end

    #Plotting Functions, "test" will plot the test data, whereas "train" will plot the train data.
    #Only PlotScatter does not use "train" or "test"
    PlotHistogram(y_test, predict_test, k, "test")
    PlotBarComparison(y_test, predict_test, k, "test")
    PlotScatter(y_train, y_test, predict_train, predict_test, k)
    PlotQQ(predict_test, k, "test")
    PlotFeatureImportance(data_plot, k)

end

