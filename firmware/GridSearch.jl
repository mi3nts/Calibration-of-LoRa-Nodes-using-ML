#Tuning through grid search

function GridSearch(mlmodel, X, y)
    
    #hyperparameters
    r1 = MLJ.range(mlmodel, :max_depth, lower = 5, upper = 20, scale = :linear)
    r2 = MLJ.range(mlmodel, :max_samples_split, lower=2, upper=5, scale = :linear)
    grid = Grid(resolution = 10)

    tuned_model = TunedModel(model = mlmodel(), tuning=grid, resampling = CV(nfolds=3), range = [r1, r2], measure=rsquared)
    mach = machine(tuned_model, X, vec(Matrix(y)))
    fit!(mach, verbosity = 0)
    println(fitted_params(mach).best_model)
    println(report(mach).best_history_entry)

end