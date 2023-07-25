RandomForestRegressor = @load RandomForestRegressor pkg=MLJScikitLearnInterface verbosity = 0
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree verbosity = 0
NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux verbosity = 0
ExtraTreesRegressor = @load ExtraTreesRegressor pkg=MLJScikitLearnInterface verbosity = 0
XGBoostRegressor = @load XGBoostRegressor pkg=XGBoost verbosity = 0
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels verbosity = 0
EvoTreeRegressor = @load EvoTreeRegressor pkg=EvoTrees verbosity = 0
LGBMRegressor = @load LGBMRegressor pkg=LightGBM verbosity = 0
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity = 0


function SuperLearner(k, X_train, y_train, X_test, y_test, wholedata)

    #grimm_dictionary with models
    #grimm_dict = Dict("nnr" => NeuralNetworkRegressor(builder=MLJFlux.MLP(hidden=(128,64), σ = Flux.σ), optimiser=Flux.ADAM(0.001), loss=Flux.mse, epochs=16, batch_size=6, rng=StableRNG(42)),
    #"dtr" => DecisionTreeRegressor(), "edtr" => EnsembleModel(model=dtr, n=100, bagging_fraction=0.8), "rfr" => RandomForestRegressor(n_trees=52), "xgbr" => XGBoostRegressor(max_depth = 5, num_round = 14),
    #"knnr" => KNNRegressor(K = 27, algorithm = :kdtree, leafsize = 21), "etr" => EvoTreeRegressor(max_depth = 9, nrounds=56), "lgbr" => LGBMRegressor(num_leaves = 41, min_data_in_leaf = 1),
    #"extra" => ExtraTreesRegressor(max_depth = 7, n_estimators=93), "rr" => RidgeRegressor(lambda = 0.010000000000000004, scale_penalty_with_samples = false))

    #palas_dictionary
    #palas_dict = Dict("nnr" => NeuralNetworkRegressor(builder=MLJFlux.MLP(hidden=(128,64), σ = Flux.σ), optimiser=Flux.ADAM(0.001), loss=Flux.mse, epochs=16, batch_size=6, rng=StableRNG(42)),
    #"dtr" => DecisionTreeRegressor(), "edtr" => EnsembleModel(model=dtr, n=100, bagging_fraction=0.8), "rfr" => RandomForestRegressor(n_trees=52), "xgbr" => XGBoostRegressor(max_depth = 5, num_round = 14),
    #"knnr" => KNNRegressor(K = 27, algorithm = :kdtree, leafsize = 21), "etr" => EvoTreeRegressor(max_depth = 9, nrounds=56), "lgbr" => LGBMRegressor(num_leaves = 41, min_data_in_leaf = 1),
    #"extra" => ExtraTreesRegressor(max_depth = 7, n_estimators=93), "rr" => RidgeRegressor(lambda = 0.010000000000000004, scale_penalty_with_samples = false))

    #Neuralnetworkregressor
    nnr = NeuralNetworkRegressor(builder=MLJFlux.MLP(hidden=(128,64), σ = Flux.σ), optimiser=Flux.ADAM(0.001), loss=Flux.mse, epochs=16, batch_size=6, rng=StableRNG(42))

    #decision tree
    dtr = DecisionTreeRegressor()

    #Ensemble of Trees
    edtr = EnsembleModel(model=dtr, n=100, bagging_fraction=0.8)

    #random forest regressor
    rfr = RandomForestRegressor()

    #xgboost
    xgbr = XGBoostRegressor(max_depth = 5, num_round = 14)

    #KNNRegressor
    knnr = KNNRegressor(K = 27, algorithm = :kdtree, leafsize = 21)

    #Evotree regressor
    etr = EvoTreeRegressor(max_depth = 9, nrounds=56)

    #LGBMRegressor
    lgbr = LGBMRegressor(num_leaves = 41, min_data_in_leaf = 1)

    #ExtraTreesRegressor
    extra=ExtraTreesRegressor(max_depth = 7, n_estimators=93)

    #RidgeRegressor
    rr = RidgeRegressor(lambda = 0.010000000000000004, scale_penalty_with_samples = false)
    
    #defining stack
    stack = MLJ.Stack(;metalearner = nnr,
        resampling = CV(nfolds=3, shuffle=true, rng=123),
        measures=rsquared,
        nnr=nnr,
        dtr=dtr,
        edtr=edtr,
        rfr=rfr,
        xgbr=xgbr,
        knnr=knnr,
        lgbr=lgbr,
        extra=extra,
        rr=rr,
        )
    
    # Training model
    model = machine(stack, X_train, vec(Matrix(y_train)))
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
    println("test r2 value for " * k * " = " * string(r2_score_test))
    #mse_test = round(mse(predict_test, Matrix(y_test)), digits=3)
    #println("test mse value for " * k * " = " * string(mse_test))
    #rmse_test = round(sqrt(mse_test), digits=3)
    #println("test rmse value for " * k * " = " * string(rmse_test))
    
    
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


