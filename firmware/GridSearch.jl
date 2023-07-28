#Tuning through grid search

function GridSearch(mlmodel, X, y)
    
    #hyperparameters
    r1 = MLJ.range(mlmodel, :max_depth, lower=14, upper=23, scale=:linear)  
    #r2 = MLJ.range(Int, :min_samples_leaf, lower=2, upper=6, scale=:linear)  
    grid = Grid(resolution=9)
    tuned_model = TunedModel(model = mlmodel, tuning=grid, resampling = Holdout(fraction_train=0.8), range = [r1], measure=rsquared)
    mach = machine(tuned_model, X, vec(Matrix(y)))
    fit!(mach, verbosity = 0)
    display(fitted_params(mach).best_model)

end