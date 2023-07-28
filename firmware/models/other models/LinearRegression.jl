LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface verbosity = 0

# Linear Regression Function

function LinearRegression(k, X_train, y_train, X_test, y_test, wholedata)

    # Training model
    model = machine(LinearRegressor(), X_train, vec(Matrix(y_train)))
    MLJ.fit!(model, verbosity = 0)
    predict_train = MLJ.predict(model, X_train)
    predict_test = MLJ.predict(model, X_test)
    # ------------------ Cross Validation ---------------------#
    k_fold = 5
    X = vcat(X_train, X_test)
    y = vcat(y_train, y_test)

    #implement grid search
    #GridSearch(Regressor(max_depth=20, n_trees = 50), X, y)
    
    #K-fold crossvalidatoin
    #KFoldCV(X, y, k_fold, k, Regressor())

    #Print r2, mse, and rmse values for test data
    r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    println("Linear Regression: test r2 value for " * k * " = " * string(r2_score_test))
    #mse_test = round(mse(predict_test, Matrix(y_test)), digits=3)
    #println("Linear Regression: test mse value for " * k * " = " * string(mse_test))
    #rmse_test = round(sqrt(mse_test), digits=3)
    #println("Linear Regression: test rmse value for " * k * " = " * string(rmse_test))
    
    
    # Calculating Feature Importance using the FeatureImportance Function from FeatureImportance.jl
    data_plot = FeatureImportance(wholedata, k, model)
    
    #copying the target variable name before changing latex Formatting
    kcopy = k
    
    #LaTex Formatting
    if k[1:2] == "pm" && k != "pmTotal"
        if occursin(k, "pm2_5")
            k = replace(k, "_" => ".")
        end
        k = "PM" * latexstring("_{" * k[3:length(k)] * "}") * " (μg/m³)"
    elseif k == "pmTotal"
        k = "Total PM (μg/m³)"
    elseif k == "alveolic"
        k = "Alveolic (μg/m³)"
    elseif k == "thoracic"
        k = "Thoracic (μg/m³)"
    elseif k == "inhalable"
        k = "Inhalable (μg/m³)"
    elseif k == "dCn" 
        k = "dCn (#/cm³)"
    end
    

    #Plotting Functions
    PlotHistogram(y_test, predict_test, k, kcopy)
    PlotBarComparison(y_test, predict_test, k, kcopy)
    PlotScatter(y_train, y_test, predict_train, predict_test, k, kcopy)
    PlotQQ(y_test, predict_test, k, kcopy)
    PlotFeatureImportance(data_plot, k, kcopy)

end



