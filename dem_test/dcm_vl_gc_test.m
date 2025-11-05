


opts.gc_order   = 0;                 % start simple
opts.logpsd     = false;             % per your preference
opts.mask_fun   = @(f) f>1 & f<120;
opts.beta_sched = [0.2 0.5 1.0];
opts.maxiter    = 128;
opts.step       = [];                % (ignored now; LM handles size)
opts.plot       = true; 
opts.plot_every = 5;

DCM = dcm_vl_gc(DCM, opts);