RandomForestRegressor = @load RandomForestRegressor pkg=MLJScikitLearnInterface verbosity = 0
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=BetaML verbosity = 0
LinearRegressor = @load LinearRegressor pkg=MLJScikitLearnInterface verbosity = 0
ExtraTreesRegressor = @load ExtraTreesRegressor pkg=MLJScikitLearnInterface verbosity = 0

function SuperLearner(X_train, y_train, X_test, y_test)
    stack = MLJ.Stack(;metalearner = ExtraTreesRegressor(),
        resampling = CV(nfolds=5, shuffle=true, rng=123),
        measures=rsquared,
        random_forest = RandomForestRegressor(),
        tree = DecisionTreeRegressor(),
        extra = ExtraTreesRegressor())
    
    mach = machine(stack, X_train, vec(Matrix(y_train)))
    MLJ.fit!(mach, verbosity = 0)
    println("----superlearner-------")
    println(r2_score(MLJ.predict(mach, X_test), Matrix(y_test)))
    #println(evaluate!(rfr, measure=rsquared))

end