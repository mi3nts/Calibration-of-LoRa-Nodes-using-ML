RandomForestRegressor = @load RandomForestRegressor pkg=MLJDecisionTreeInterface verbosity = 0
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=MLJDecisionTreeInterface verbosity = 0
NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux verbosity = 0
ExtraTreesRegressor = @load ExtraTreesRegressor pkg=MLJScikitLearnInterface verbosity = 0
XGBoostRegressor = @load XGBoostRegressor pkg=MLJXGBoostInterface verbosity = 0
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels verbosity = 0
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees verbosity = 0
LGBMRegressor = @load LGBMRegressor pkg=LightGBM verbosity = 0
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity = 0


rng = StableRNG(42)
function SuperLearner(k, X_train, y_train, X_test, y_test, wholedata)

    #Neural network regressor
    nnr = NeuralNetworkRegressor(builder=MLJFlux.MLP(hidden=(250,100,50), σ=NNlib.relu),
    batch_size = 200,
    optimiser=Flux.Optimise.ADAM(0.001),
    lambda = 0.0001,  
    rng=rng,
    epochs=200)

    #decision tree
    dtr = DecisionTreeRegressor()

    #Ensemble of Trees
    edtr = EnsembleModel(model=dtr, n=100, bagging_fraction=bf)

    #random forest regressor
    rfr = RandomForestRegressor()

    #xgboost
    xgbr = XGBoostRegressor()

    #KNNRegressor
    knnr = KNNRegressor()

    #Evotree regressor
    etr = EvoTreeRegressor()

    #LGBMRegressor
    lgbr = LGBMRegressor()

    #defining stack

    stack = MLJ.Stack(;metalearner = RidgeRegressor(),
        resampling = CV(nfolds=5, shuffle=true, rng=123),
        measures=rsquared,
        nnr=nnr,
        dtr=dtr,
        edtr=edtr,
        rfr=rfr,
        xgbr=xgbr,
        knnr=knnr,
        etr=etr,
        lgbr=lgbr
        )
    
    # Training model
    model = machine(stack, X_train, vec(Matrix(y_train)))
    MLJ.fit!(model, verbosity = 0)
    predict_train = MLJ.predict(model, X_train)
    predict_test = MLJ.predict(model, X_test)
    
    println("----superlearner-------")
    println(r2_score(MLJ.predict(model, X_test), Matrix(y_test)))
    # ------------------ Cross Validation ---------------------#
    k_fold = 5
    X = vcat(X_train, X_test)
    y = vcat(y_train, y_test)

    #implement grid search
    #GridSearch(LinearRegressor(), X, y)
    
    #K-fold crossvalidation
    #KFoldCV(X, y, k_fold, k, stack)

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
    #PlotBarComparison(y_test, predict_test, k, kcopy)
    PlotScatter(y_train, y_test, predict_train, predict_test, k, kcopy)
    PlotQQ(y_test, predict_test, k, kcopy)
    PlotFeatureImportance(data_plot, k, kcopy)
    

end



