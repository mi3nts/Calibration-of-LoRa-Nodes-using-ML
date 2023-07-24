#Tuning through grid search

function GridSearch(mlmodel, X, y)
    
    #hyperparameters
    r1 = MLJ.range(mlmodel, :min_samples_split, lower=1, upper=10, scale = :linear)
    r2 = MLJ.range(mlmodel, :min_samples_leaf, lower=1, upper=10, scale = :linear)
    grid = Grid(resolution = 3)
    tuned_model = TunedModel(model = mlmodel, tuning=grid, resampling = CV(nfolds=3), range = [r1, r2], measure=rsquared)
    mach = machine(tuned_model, X, vec(Matrix(y)))
    fit!(mach, verbosity = 0)
    display(fitted_params(mach).best_model)

end