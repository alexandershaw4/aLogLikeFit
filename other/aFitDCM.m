classdef aFitDCM < handle
% An object wrapper for inverting Spectral Dynamic Causal Models using AO.m 
% optimisation routine.    
% 
% Example usage:
%     m = AODCM(DCM);   %<-- construct the object and autopopulate fields
%     m.aloglik(n);     %<-- run the optimisation 
%
%
% AS2020

    properties
        
        DCM     % the full spec'd DCM structure
        pE      % reduced priors, based on DCM.M.pE & DCM.M.pC
        pC      % reduced prior variances from DCM.M.pC
        opts    % t
        X       % posterior values resulting from optim
        Ep      % same as X but in structured space
        F       % posterior objective function value (deflt: Free Energy)
        CP      % posterior parameter covariance in reduced space
        history % history from the optimisation (steps, partial derivatives)
        DD      % a helper structure for the embedded wrapdm function
        Y       % the stored data from DCM.xY.y
        V       % maps between full (parameter) space and reduced
        iserp   % switch to ERP models rather than spectral
        ftype   % swith between reduced and full model 
        hist
        params
        Pp
        FreeEnergyF
        iS
        D
        allp
        histp
        VV
    end
    
    methods
                function obj = update_parameters(obj,P)
            % after contruction, allow updating object priors
            P        = spm_unvec( spm_vec(P), obj.DCM.M.pE);
            obj.DD.P = spm_vec(P);
            
            % also save the optimisation hisotry structure from each call
            % to the optimimser
            try obj.hist = [obj.hist; obj.history];
            catch obj.hist = obj.history;
            end

            try obj.hist = [obj.hist {obj.allp}];
            catch
                try obj.hist = {obj.allp};
                
                end
            end
        end
        
        function obj = aFitDCM(DCM,ftype)
            % Class constructor - initates the options structure for the
            % optimisation
            obj.DCM = DCM;
            
            if nargin > 1 && ~isempty(ftype)
                % wrapped function
                obj.ftype = ftype;
            else
                % straight function
                obj.ftype = 1;
            end
            
            DD    = obj.DCM;
            DD.SP = obj.DCM.M.pE;
            P     = spm_vec(obj.DCM.M.pE);
            V     = spm_vec(obj.DCM.M.pC);
            
            % Create mapping (cm) between full and reduced space
            cm = spm_svd(diag(V),0);
            ip = find(V);
            
            if obj.ftype == 1
                % to pass to f(ßx)
                DD.P  = P;
                DD.V  = V;
                DD.cm = cm;

                % Reduced parameter vectors -
                p = ones(length(ip),1);
                c = V(ip);

                opts.x0 = p;
                opts.V = V(ip);
                
            else
                DD.P  = P;
                DD.V  = V;
                DD.cm = eye(length(P));
                p     = P;
                c     = V;
                
                % Essential inputs for optimisation
                opts     = struct;
                opts.fun = @(p) spm_vec( feval(DCM.M.IS,spm_unvec(p,DCM.M.pE),DCM.M,DCM.xU) );
                opts.x0  = p(:);
                opts.V   = c(:); 
            end
            
            
            opts.y   = spm_vec(obj.DCM.xY.y);
            opts.y = [real(opts.y); imag(opts.y)];
           
                       
            % save this read for inversion
            obj.opts = opts;
            obj.pE   = p(:);
            obj.pC   = c(:);
            obj.DD   = DD;
            obj.Y    = [real(DCM.xY.y{1}); imag(DCM.xY.y{1})];
            
            
        end

        function [y,PP,s,t,centrefreqs] = wrapdm(obj,Px,varargin)
            % wraps the DCM/SPM integrator function into a f(P)
            % anonymous-like function accepting a reduced parameter vector
            % and returning the model output
            %
            % Constructs the model:
            %     log( M.V*M.X.*exp(M.DD.P) ) == M.V'*M.Ep
            %
            % so that AO.m actually optimises X
            s=[];t=[];
            DD   = obj.DD;
            P    = DD.P;
            cm   = DD.cm;
            
            X0 = cm*Px(:);
            X0(X0==0)=1;
            X0 = full(full(X0).*exp(full(P(:))));
            X0 = log(X0);
            X0(isinf(X0)) = 0;
            
            PP = spm_unvec(X0,DD.SP);
            
            if isfield(PP,'J')
               % neural masses with a J parameter
               PP.J(PP.J==0)=-1000;
            end
            
            IS   = spm_funcheck(DD.M.IS);       % Integrator

            % if nargin == 3 
            %     if ~isstruct(varargin{1}) && varargin{1} == 1
            %         % trigger fixed-point search
            %         x0 = atcm.fun.alexfixed(PP,DD.M,1e-6);
            % 
            %         obj.DD.M.x = spm_unvec(x0,obj.DD.M.x);
            % 
            %     end
            % 
            % end
            
            %if nargout(IS) < 8
            %generic, works for all functions....
                y    = IS(PP,DD.M,DD.xU);
 
           %  elseif nargout(IS) == 8
           %  % this is specific to atcm.integrate3.m
           %      [y,w,s,g,t,pst,l,oth] = IS(PP,DD.M,DD.xU);
           %      s = (s{1});
           %      s = reshape(s,[size(s,1)*size(s,2)*size(s,3),size(s,4)]);
           %      jj = find(exp(PP.J));
           %      s = s(jj,:);
           %      t = pst;
           %     % centrefreqs = l{1}.centrals{1};
           %     centrefreqs=[];
           % end
            
            %y    = IS(PP,DD.M,DD.xU);           % Prediction
            y    = spm_vec(y);
            y    = real(y);

            %y = [real(y); imag(y)];
            
        end

        function obj = nlls_optimise(obj)
            
            options = statset;
            options.Display = 'iter';
            options.TolFun  = 1e-6;
            options.MaxIter = 32;
            options.FunValCheck = 'on';
            options.DerivStep = 1e-8;

            funfun = @(b,p) full(spm_vec(spm_cat(obj.opts.fun(b.*p))));
            
            [BETA,R,J,COVB,MSE] = atcm.optim.nlinfit(obj.opts.x0,...
                            full(spm_vec(spm_cat(obj.opts.y))),funfun,full(obj.opts.x0),options);
           
            [ff,pp]=obj.opts.fun(BETA.*obj.opts.x0);
            
            obj.X = spm_vec(pp);
            
                 
            %obj.X  = obj.V*(BETA.*obj.opts.x0);
            %obj.Ep = spm_unvec(spm_vec(obj.X),obj.DD.P);
            obj.Ep = spm_vec(obj.X);
            obj.CP = COVB;
            obj.F  = MSE;
            
        end

        function afitNN(obj)
        
             x0  = obj.opts.x0(:);
            fun = @(varargin)obj.wrapdm(varargin{:});

            %x0 = spm_vec(obj.DCM.M.pE);
            M  = obj.DCM.M;
            %V  = spm_vec(obj.DCM.M.pC);
            V  = (obj.opts.V );
            y  = spm_vec(obj.DCM.xY.y);


            % purturb the system a bunch of times and get outputs
            fprintf('Generating test data: simulating\n');
            [X_train, Y_train] = generate_training_data(fun, x0, V, 1000);

            % Train Neural Network
            fprintf('Training fully-connected NN\n');
            net = train_nn(Y_train, X_train);

            % accuracy recovering priors
            x_true = x0;
            y_empirical = fun(x_true); % Replace with real experimental spectral data
            x_estimated = infer_parameters(net, y_empirical(:)');
            y_predicted = fun(double(x_estimated(:)));

            % recovering posteriors
            x_estimated = infer_parameters(net, y(:)');
            y_predicted = fun(double(x_estimated(:)));

            % Plot Example Spectra
            figure; w = 1:length(y_empirical);
            plot(w, y, 'b',w,y_predicted,'r', 'LineWidth', 2);
            title('Model Fit');
            xlabel('Frequency Bin');
            ylabel('Power');
            grid on;


            % return outputs
            obj.X = double(x_estimated(:));
            obj.F = sum( (y(:) - y_predicted(:)).^2);
            [~, P] = fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);

        end

        function aloglik(obj,maxit)

            if nargin < 2; 
                maxit = 32;
            end

            %fun = @(P,M) spm_vec(obj.DCM.M.IS(spm_unvec(P,obj.DCM.M.pE),obj.DCM.M,obj.DCM.xU));

            x0  = obj.opts.x0(:);
            fun = @(varargin)obj.wrapdm(varargin{:});

            %x0 = spm_vec(obj.DCM.M.pE);
            M  = obj.DCM.M;
            %V  = spm_vec(obj.DCM.M.pC);
            V  = (obj.opts.V );
            y  = spm_vec(obj.DCM.xY.y);

            sigma = 1 * ones(size(y));

            %[x_est, logL, iter] = fitLogLikelihoodGN(y, fun, x0, sigma, maxit, 1e-6);

            [obj.X, obj.F, iter,obj.CP,obj.allp] = fitLogLikelihoodLM(y, fun, x0, V, sigma, maxit, 1e-6, 0.1);


            [~, P] = fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);

        end

        

        function aloglikVL(obj,maxit)

            if nargin < 2; 
                maxit = 32;
            end

            %fun = @(P,M) spm_vec(obj.DCM.M.IS(spm_unvec(P,obj.DCM.M.pE),obj.DCM.M,obj.DCM.xU));

            x0  = obj.opts.x0(:);
            fun = @(varargin)obj.wrapdm(varargin{:});

            %x0 = spm_vec(obj.DCM.M.pE);
            M  = obj.DCM.M;
            V  = diag(obj.opts.V );
            y  = spm_vec(obj.DCM.xY.y);

            [obj.X, obj.CP, obj.F] = fitVariationalLaplace(y, fun, x0, V, maxit, 1e-6);

            [~, P] = fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);

        end

        function aloglikVLtherm(obj,maxit,plots)

            if nargin < 2 || isempty(maxit)
                maxit = 32;
            end

            if nargin < 3
                plots = 1;
            end

            %fun = @(P,M) spm_vec(obj.DCM.M.IS(spm_unvec(P,obj.DCM.M.pE),obj.DCM.M,obj.DCM.xU));

            x0  = obj.opts.x0(:);
            fun = @(varargin)obj.wrapdm(varargin{:});

            %x0 = spm_vec(obj.DCM.M.pE);
            M  = obj.DCM.M;
            V  = diag(obj.opts.V );
            y  = spm_vec(obj.DCM.xY.y);%[real(spm_vec(obj.DCM.xY.y)); imag(spm_vec(obj.DCM.xY.y))];

            % [m, V, D, logL, iter, sigma2, allm] 
            [obj.X, obj.VV, obj.D, obj.F,~,~,obj.allp] = fitVariationalLaplaceThermo(y, fun, x0, V, maxit, 1e-6,plots);
            %[obj.X, obj.CP, obj.F] = fitVariationalLaplaceThermo4thOrder(y, fun, x0, V, maxit, 1e-6);
            %[obj.X, obj.CP, obj.F] = fitVariationalLaplaceNF(y, fun, x0, V, maxit, 1e-6);

            [~, P] = fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);
            %obj.V = obj.CP;
            %obj.CP = obj.CP * obj.CP' + obj.D;
            
            %V = obj.VV;
            
            obj.CP = pinv( (obj.VV*obj.VV') + obj.D);

            % Ainv = diag(1 ./ diag(obj.D));
            % M = eye(size(V,2)) + V' * Ainv * V;
            % invM = inv(M);  % small matrix
            % obj.CP = Ainv - Ainv * V * invM * V' * Ainv;


        end

        function aloglikVLtherm_cov(obj,maxit,plots)

            if nargin < 2 || isempty(maxit)
                maxit = 32;
            end

            if nargin < 3
                plots = 1;
            end

            %fun = @(P,M) spm_vec(obj.DCM.M.IS(spm_unvec(P,obj.DCM.M.pE),obj.DCM.M,obj.DCM.xU));

            x0  = obj.opts.x0(:);
            fun = @(varargin)obj.wrapdm(varargin{:});

            %x0 = spm_vec(obj.DCM.M.pE);
            M  = obj.DCM.M;
            V  = diag(obj.opts.V );
            y  = spm_vec(obj.DCM.xY.y);%[real(spm_vec(obj.DCM.xY.y)); imag(spm_vec(obj.DCM.xY.y))];

            % [m, V, D, logL, iter, sigma2, allm] 
            [obj.X, obj.VV, obj.D, obj.F,~,~,obj.allp] = fitVL_LowRankNoise(y, fun, x0, V, maxit, 1e-6,plots);
            %[obj.X, obj.CP, obj.F] = fitVariationalLaplaceThermo4thOrder(y, fun, x0, V, maxit, 1e-6);
            %[obj.X, obj.CP, obj.F] = fitVariationalLaplaceNF(y, fun, x0, V, maxit, 1e-6);

            [~, P] = fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);

            obj.CP = pinv( (obj.VV*obj.VV') + obj.D);

           % obj.V = obj.CP;
           % obj.CP = obj.CP * obj.CP' + obj.D;
            
            % V = obj.VV;
            % 
            % Ainv = diag(1 ./ diag(obj.D));
            % M = eye(size(V,2)) + V' * Ainv * V;
            % invM = inv(M);  % small matrix
            % obj.CP = Ainv - Ainv * V * invM * V' * Ainv;

        end

        function fitRL(obj,maxit)

            if nargin < 2 
                maxit = 100;
            end

            x0  = obj.opts.x0(:);
            fun = @(varargin)obj.wrapdm(varargin{:});

            %x0 = spm_vec(obj.DCM.M.pE);
            M  = obj.DCM.M;
            V  = diag(obj.opts.V );
            y  = spm_vec(obj.DCM.xY.y);

            [obj.X, obj.F] = rl_parameter_optimization(fun, x0, y, V,maxit);

            [~, P] = fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);
        end


        % function aloglikp(obj,maxit)
        % 
        %     if nargin < 2; 
        %         maxit = 32;
        %     end
        % 
        %     %fun = @(P,M) spm_vec(obj.DCM.M.IS(spm_unvec(P,obj.DCM.M.pE),obj.DCM.M,obj.DCM.xU));
        % 
        %     x0  = obj.opts.x0(:);
        %     fun = @(varargin)obj.wrapdm(varargin{:});
        % 
        %     %x0 = spm_vec(obj.DCM.M.pE);
        %     M  = obj.DCM.M;
        %     %V  = spm_vec(obj.DCM.M.pC);
        %     y  = spm_vec(obj.DCM.xY.y);
        % 
        %     sigma = 1 * ones(size(y));
        % 
        %     %[x_est, logL, iter] = fitLogLikelihoodGN(y, fun, x0, sigma, maxit, 1e-6);
        % 
        %     num_basis = 12;
        % 
        %     [obj.X, obj.F, iter,obj.CP] = fitLogLikelihoodLMprecision(y, fun, x0, sigma, maxit, 1e-6, 0.1,num_basis);
        % 
        %     [~, P] = fun(spm_vec(obj.X));
        %     obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);
        % 
        % end

        function aloglikFE(obj,maxit)

            if nargin < 2; 
                maxit = 32;
            end

            %fun = @(P,M) spm_vec(obj.DCM.M.IS(spm_unvec(P,obj.DCM.M.pE),obj.DCM.M,obj.DCM.xU));

            x0  = obj.opts.x0(:);
            fun = @(varargin)obj.wrapdm(varargin{:});

            %x0 = spm_vec(obj.DCM.M.pE);
            M  = obj.DCM.M;
            %V  = spm_vec(obj.DCM.M.pC);
            y  = spm_vec(obj.DCM.xY.y);
            V   = obj.opts.V;
            stv = sqrt(V);

            sigma = 2 * ones(size(y));

            %[x_est, logL, iter] = fitLogLikelihoodGN(y, fun, x0, sigma, maxit, 1e-6);

            %[obj.X, obj.F, iter,obj.CP] = fitLogLikelihoodLM(y, fun, x0, sigma, maxit, 1e-6, 0.1)

            [obj.X, obj.F, iter,obj.CP] = fitLogLikelihoodLMFE(y, fun, x0, sigma, maxit, 1e-6, 0.1,stv)

            [~, P] = fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);

        end

        % function aFreeEnergyLM(obj,maxit)
        % 
        %     if nargin < 2; 
        %         maxit = 32;
        %     end
        % 
        %     %fun = @(P,M) spm_vec(obj.DCM.M.IS(spm_unvec(P,obj.DCM.M.pE),obj.DCM.M,obj.DCM.xU));
        % 
        %     x0  = obj.opts.x0(:);
        %     V   = obj.opts.V;
        %     fun = @(varargin)obj.wrapdm(varargin{:});
        % 
        %     %x0 = spm_vec(obj.DCM.M.pE);
        %     M  = obj.DCM.M;
        %     %V  = spm_vec(obj.DCM.M.pC);
        %     y  = spm_vec(obj.DCM.xY.y);
        % 
        %     sigma = ones(size(y))./length(y);
        % 
        %     %[x_est, logL, iter] = fitLogLikelihoodGN(y, fun, x0, sigma, maxit, 1e-6);
        % 
        %     % y, f, x0, sigma, mu_prior, sigma_prior, maxIter, tol, lambda0)
        %     [obj.X, obj.F, iter] = fitFreeEnergyLM(y, fun, x0, sigma,V, maxit, 1e-6, 0.1)
        % 
        %     [~, P] = fun(spm_vec(obj.X));
        %     obj.Ep = spm_unvec(spm_vec(P),obj.DD.M.pE);
        % 
        % end

    end

end

