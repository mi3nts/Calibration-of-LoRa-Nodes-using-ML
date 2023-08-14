RandomForestRegressor = @load RandomForestRegressor pkg=MLJScikitLearnInterface verbosity = 0
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=BetaML verbosity = 0
NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux verbosity = 0
ExtraTreesRegressor = @load ExtraTreesRegressor pkg=MLJScikitLearnInterface verbosity = 0
LGBMRegressor = @load LGBMRegressor pkg=LightGBM verbosity = 0
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity = 0
KNeighborsRegressor = @load KNeighborsRegressor pkg=MLJScikitLearnInterface verbosity=0


function SuperLearner(k, X_train, y_train, X_test, y_test, wholedata)

    #palas_dictionary
    palas_dict = Dict("nnr" => NeuralNetworkRegressor(builder=MLJFlux.MLP(hidden=(256,256,128,128), σ = Flux.relu), optimiser=Flux.ADAM(0.001), loss=Flux.mse, epochs=16, batch_size=6, rng=StableRNG(42)),
    "dtr" => DecisionTreeRegressor(), 
    "edtr" => EnsembleModel(model=DecisionTreeRegressor(), n=100, bagging_fraction=0.8), 
    "rfr" => RandomForestRegressor(),
    "knnr" => KNeighborsRegressor(n_neighbors=7, weights="distance", metric="euclidean"),
    "lgbr" => LGBMRegressor(num_iterations=160, lambda_l1 = 85.3, lambda_l2 = 6.2, min_gain_to_split=0.05, num_leaves = 114, min_data_in_leaf = 3, learning_rate=0.08),
    "extra" => ExtraTreesRegressor(n_estimators=111), 
    "rr" => RidgeRegressor(lambda = 0.010000000000000004, scale_penalty_with_samples = false)    )
    
    #defining stack
    stack = MLJ.Stack(;metalearner = palas_dict["knnr"],
        resampling = CV(nfolds=4, shuffle=true, rng=123),
        measures=rsquared,
        dtr=palas_dict["dtr"],
        edtr=palas_dict["edtr"],
        rfr=palas_dict["rfr"],    
        lgbr=palas_dict["lgbr"],
        extra=palas_dict["extra"],
        nnr=palas_dict["nnr"]
        )
    
    # scale data
    sklearn_preprocessing = pyimport("sklearn.preprocessing")
    scaler = sklearn_preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(Matrix(X_train))
    X_test_scaled = scaler.transform(Matrix(X_test))
    
    col_name = names(X_train)
    X_train = DataFrames.DataFrame(pyconvert(Matrix{Float32}, X_train_scaled), :auto)
    X_test = DataFrames.DataFrame(pyconvert(Matrix{Float32}, X_test_scaled), :auto)

    for (n, old_col) in enumerate(names(X_train))
        rename!(X_train, Symbol(old_col) => Symbol(col_name[n]))
        rename!(X_test, Symbol(old_col) => Symbol(col_name[n]))
    end


    # Training model
    model = machine(stack, X_train, vec(Matrix(y_train)))
    MLJ.fit!(model, verbosity = 0)
    predict_train = MLJ.predict(model, X_train)
    predict_test = MLJ.predict(model, X_test)
    
    # ------------------ Cross Validation ---------------------#
    k_fold = 5
    X = vcat(X_train, X_test)
    y = vcat(y_train, y_test)
    
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
    if k == "pm2_5"
        k = "PM₂.₅ (μg/m³)"
    elseif k == "pm4"
        k = "PM₄.₀ (μg/m³)"
    elseif k == "pm10"
        k = "PM₁₀.₀ (μg/m³)"
    elseif k == "pm1"
        k = "PM₁.₀ (μg/m³)"
    elseif k == "alveolic"
        k = "Alveolic (μg/m³)"
    elseif k == "inhalable"
        k = "Inhalable (μg/m³)"
    elseif k == "thoracic"
        k = "Thoracic (μg/m³)"
    elseif k == "dCn"
        k = "Particle Count Density (#/cm³)"
    elseif k == "pmTotal"
        k = "Total PM Concentration (μg/m³)"
    end
    

    #Plotting Functions
    PlotHistogram(y_test, predict_test, k, kcopy)
    PlotBarComparison(y_test, predict_test, k, kcopy)
    PlotScatter(y_train, y_test, predict_train, predict_test, k, kcopy)
    PlotQQ(y_test, predict_test, k, kcopy)
    PlotFeatureImportance(data_plot, k, kcopy)

end


