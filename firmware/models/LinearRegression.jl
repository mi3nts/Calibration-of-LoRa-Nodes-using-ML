LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface verbosity = 0

# Linear Regression Function

function LinearRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Training model
    lm = machine(LinearRegressor(), X_train, vec(Matrix(y_train)))
    MLJ.fit!(lm, verbosity = 0)
    predict_train = MLJ.predict(lm, X_train)
    predict_test = MLJ.predict(lm, X_test)


#--------------------------Cross Validation--------------------------#
    k_fold = 10
    a = collect(MLBase.Kfold(size(X_train)[1], k_fold))
    i= 1

    for i in 1:k_fold
        row = a[i]
        temp_X_train = X_train[row,:]
        temp_y_train = y_train[row,:]

        temp_X_test = X_train[setdiff(1:end, row),:]
        temp_y_test = y_train[setdiff(1:end, row),:]
        
        lm = machine(LinearRegressor(), temp_X_train, vec(Matrix(temp_y_train)))
        MLJ.fit!(lm, verbosity = 0)

        temp_predict_y_train = MLJ.predict(lm, temp_X_train)
        temp_predict_y_test = MLJ.predict(lm, temp_X_test)
    
        temp_mse_test = round(mse(temp_predict_y_test, Matrix(temp_y_test)), digits=3)
        #println("Linear Regression: test mse value for " * k * " = " * string(temp_mse_test))
        temp_rmse_test = sqrt(temp_mse_test)
        #println("Linear Regression: test rmse value for " * k * " = " * string(temp_rmse_test))
        r2_score_test = round(r2_score(temp_predict_y_test, Matrix(temp_y_test)), digits=3)
        println("R squared error for $k , fold $i is ",r2_score_test)
    end
    # #Print r2, mse, and rmse values for test data
    # #r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    # #println("Linear Regression: test r2 value for " * k * " = " * string(r2_score_test))
    # mse_test = round(mse(predict_y_test, Matrix(y_test)), digits=3)
    # println("Linear Regression: test mse value for " * k * " = " * string(mse_test))
    # rmse_test = sqrt(mse_test)
    # println("Linear Regression: test rmse value for " * k * " = " * string(rmse_test))


    # Calculating Feature Importance using the FeatureImportance Function from FeatureImportance.jl
    data_plot = FeatureImportance(wholedata, k, lm)


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


