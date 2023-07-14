RandomForestRegressor = @load RandomForestRegressor pkg=MLJScikitLearnInterface verbosity = 0
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=BetaML verbosity = 0
LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface verbosity = 0
ExtraTreesRegressor = @load ExtraTreesRegressor pkg=MLJScikitLearnInterface verbosity = 0

function SuperLearner(k, X_train, y_train, X_test, y_test, wholedata)

    #defining stack
    stack = MLJ.Stack(;metalearner = ExtraTreesRegressor(),
        resampling = CV(nfolds=5, shuffle=true, rng=123),
        measures=rsquared,
        random_forest = RandomForestRegressor(),
        tree = DecisionTreeRegressor(),
        extra = ExtraTreesRegressor())
    
    # Training model
    model = machine(stack, X_train, vec(Matrix(y_train)))
    MLJ.fit!(model, verbosity = 0)
    predict_train = MLJ.predict(model, X_train)
    predict_test = MLJ.predict(model, X_test)
    
    println("----superlearner-------")
    println(r2_score(MLJ.predict(mach, X_test), Matrix(y_test)))
    # ------------------ Cross Validation ---------------------#
    k_fold = 5
    X = vcat(X_train, X_test)
    y = vcat(y_train, y_test)

    #implement grid search
    #GridSearch(LinearRegressor(), X, y)
    
    #K-fold crossvalidatoin
    KFoldCV(X, y, k_fold, k, stack)

    #Print r2, mse, and rmse values for test data
    #=
    println("----superlearner-------")
    r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    println("Linear Regression: test r2 value for " * k * " = " * string(r2_score_test))
    mse_test = round(mse(predict_test, Matrix(y_test)), digits=3)
    println("Linear Regression: test mse value for " * k * " = " * string(mse_test))
    rmse_test = round(sqrt(mse_test), digits=3)
    println("Linear Regression: test rmse value for " * k * " = " * string(rmse_test))
    =#
    
    # Calculating Feature Importance using the FeatureImportance Function from FeatureImportance.jl
    data_plot = FeatureImportance(wholedata, k, model)
    
    #copying the target variable name before changing latex Formatting
    kcopy = k
    
    #LaTex Formatting
    
    if k[1:2] == "pm" && k != "pmTotal"
        if occursin(k, "pm2_5")
            k = replace(k, "_" => ".")
        end
        k = "PM" * latexstring("_{" * k[3:length(k)] * "}") * " (µg/m" * latexstring("^{3}") * ")"
    elseif k == "pmTotal"
        k = "Total PM"
    end
    

    #Plotting Functions
    PlotHistogram(y_test, predict_test, k, kcopy)
    PlotBarComparison(y_test, predict_test, k, kcopy)
    PlotScatter(y_train, y_test, predict_train, predict_test, k, kcopy)
    PlotQQ(y_test, predict_test, k, kcopy)
    PlotFeatureImportance(data_plot, k, kcopy)
    

end



