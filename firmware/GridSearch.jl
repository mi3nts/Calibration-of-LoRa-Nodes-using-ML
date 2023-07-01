#Tuning through grid search

function GridSearch(mlmodel, X, y)
    
    #=
    r1 = range(mlmodel, :, lower = 1, upper = 20, scale = :linear)
    grid = Grid(resolution = 10)

    tuned_model = TunedModel(model = mlmodel(), tuning=grid, resampling = CV(nfolds=3), range = [r1], measure=rsquared)
    mach = machine(tuned_model, X, vec(Matrix(y)))
    fit!(mach, verbosity = 0)
    println(fitted_params(mach).best_model)
    =#

end