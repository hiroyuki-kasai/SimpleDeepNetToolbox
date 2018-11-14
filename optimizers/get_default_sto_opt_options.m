function options = get_default_sto_opt_options()

    options.stepsizefun     = @stepsize_alg;
    options.step_alg        = 'fix';
    options.step_init       = 0.01;
    options.lambda          = 0.1;    

    % SARAH
    options.sarah_plus      = 0;
    options.sarah_gamma     = 1;    

end

