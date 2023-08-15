RandomForestRegressor = @load RandomForestRegressor pkg=MLJScikitLearnInterface verbosity = 0
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=BetaML verbosity = 0
NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux verbosity = 0
ExtraTreesRegressor = @load ExtraTreesRegressor pkg=MLJScikitLearnInterface verbosity = 0
LGBMRegressor = @load LGBMRegressor pkg=LightGBM verbosity = 0
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity = 0
LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface verbosity = 0
KNeighborsRegressor = @load KNeighborsRegressor pkg=MLJScikitLearnInterface verbosity=0
GaussianProcessRegressor = @load GaussianProcessRegressor pkg=MLJScikitLearnInterface verbosity =0



function Regression(k, X_train, y_train, X_test, y_test, wholedata)
    if k != "pmTotal"
        return
    end

    #Regressor Models - access dictionary by key to use model. - current tuning.
    palas_dict = Dict("nnr" => NeuralNetworkRegressor(builder=MLJFlux.MLP(hidden=(92,164,128,92), σ = Flux.elu), optimiser=Flux.ADAM(0.001), loss=Flux.mse, epochs=32, batch_size=3, rng=StableRNG(42), lambda=10),
    "dtr" => DecisionTreeRegressor(), 
    "edtr" => EnsembleModel(model=DecisionTreeRegressor(), n=100, bagging_fraction=0.8), 
    "rfr" => RandomForestRegressor(),
    "knnr" => KNeighborsRegressor(n_neighbors=8, weights="distance", metric="manhattan", leaf_size=5),
    "lgbr" => LGBMRegressor(num_iterations=160, lambda_l1 = 85.3, lambda_l2 = 6.2, min_gain_to_split=0.05, num_leaves = 114, min_data_in_leaf = 3, learning_rate=0.08),
    "extra" => ExtraTreesRegressor(n_estimators=111), 
    "rr" => RidgeRegressor(lambda = 0.010000000000000004, scale_penalty_with_samples = false),
    "linear" => LinearRegressor(),
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
    model = machine(palas_dict["nnr"], X_train, vec(Matrix(y_train)))
    MLJ.fit!(model, verbosity = 0)
    predict_train = MLJ.predict(model, X_train)
    predict_test = MLJ.predict(model, X_test)
    
    # ------------------ Cross Validation ---------------------#
    k_fold = 5
    X = vcat(X_train, X_test)
    y = vcat(y_train, y_test)

    
    #K-fold crossvalidatoin
    #KFoldCV(X, y, k_fold, k, palas_dict["rfr"])


    #implement grid search
    #GridSearch(NeuralNetworkRegressor(builder=MLJFlux.MLP(hidden=(128,256,128,64), σ = Flux.elu), optimiser=Flux.ADAM(0.001), loss=Flux.mse, epochs=32, batch_size=3, rng=StableRNG(42)), X, y)

    #Print r2, mse, and rmse values for test data
    r2_score_test = round(r2_score(predict_train, Matrix(y_train)), digits=3)
    println("train r2 value for " * k * " = " * string(r2_score_test))
    r2_score_test = round(r2_score(predict_test, Matrix(y_test)), digits=3)
    println("test r2 value for " * k * " = " * string(r2_score_test))
    #mse_test = round(mse(predict_test, Matrix(y_test)), digits=3)
    #println("test mse value for " * k * " = " * string(mse_test))
    #rmse_test = round(sqrt(mse_test), digits=3)
    #println("test rmse value for " * k * " = " * string(rmse_test))
    
    # Calculating Feature Importance using the FeatureImportance Function from FeatureImportance.jl
    #=
    data_plot = FeatureImportance(wholedata, k, model)
    
    #copying the target variable name before changing latex Formatting
    kcopy = k
    
    #New Formatting
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
    
    #old LaTex Formatting
    #=if k[1:2] == "pm" && k != "pmTotal"
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
    end=#
    

    #Plotting Functions
    PlotHistogram(y_test, predict_test, k, kcopy)
    PlotBarComparison(y_test, predict_test, k, kcopy)
    PlotScatter(y_train, y_test, predict_train, predict_test, k, kcopy)
    PlotQQ(y_test, predict_test, k, kcopy)
    PlotFeatureImportance(data_plot, k, kcopy)
    =#
end



