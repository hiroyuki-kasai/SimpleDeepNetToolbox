function options = get_default_trainer_options()

    %options.stepsizefun     = @stepsize_alg;
    options.opt_alg         = 'SGD';
    options.step_alg        = 'fix';
    options.step_init       = 0.01;
    options.lambda          = 0.1;    
    options.tol_optgap      = 1.0e-12;
    options.batch_size      = 100;
    options.max_epoch       = 20;
    options.w_init          = [];
    options.f_opt           = -Inf;
    options.permute_on      = 1;
    options.verbose         = 1;
    options.store_w         = false;
    options.store_subinfo   = false;
    options.inner_repeat    = 1;

end

