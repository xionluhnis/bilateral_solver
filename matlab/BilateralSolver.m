classdef BilateralSolver
    %BILATERALSOLVER Fast bilateral solver
    %
    %SEE
    %   https://github.com/spillai/bilateral_solver
    
    properties
        grid
        lambda = 0
        Dn  = []
        Dm  = []
    end
    
    methods
        function self = BilateralSolver(grid, lambda, iterations)
            if nargin < 3
                iterations = [];
            end
            self.grid = grid;
            self.lambda  = lambda;
            [self.Dn, self.Dm] = bistochastize(grid, iterations);
        end
        
        function [xhat, info] = solve(self, x, w, tol, maxiter)
            %SOLVE(x, w, tol, maxiter)
            %
            % INPUT
            %   - x         (h,w,N)     target image to improve
            %   - w         (h,w)       confidence image
            %   - tol       (1,1)       tolerance constant
            %   - maxiter   (1,1)       maximum number of iterations
            %
            if nargin < 5
                maxiter = [];
            end
            if nargin < 4
                tol = [];
            end
            if nargin < 3 || isempty(w)
                [h, w, ~] = size(x);
                w = ones(h, w);
            end
            if ~isfloat(x)
                x = im2double(x);
            end
            if self.grid.numpy
                x = permute(x, [2, 1, 3]);
                w = permute(w, [2, 1, 3]);
            end
            x_dim = size(x);
            x = reshape(x, self.grid.npixels, []);
            w = reshape(w, self.grid.npixels, []);
            
            N = self.grid.nvertices;
            A_smooth = self.Dm - self.Dn * self.grid.blur(self.Dn);
            A_data = sparse(1:N, 1:N, self.grid.splat(w));
            A = self.lambda * A_smooth + A_data;
            xw = bsxfun(@times, x, w);
            b = self.grid.splat(xw);
            % Use simple Jacobi preconditioner
            M = sparse(1:N, 1:N, 1 ./ diag(A));
            I = sparse(1:N, 1:N, ones(1, N));
            % Flat initialization
            y0 = bsxfun(@rdivide, self.grid.splat(xw), self.grid.splat(w));
            yhat = zeros(size(y0));
            for d = 1:size(x, 2)
                [yhat(:, d), info] = pcg(A, b(:, d), tol, maxiter, M, I, y0(:, d));
            end
            xhat = self.grid.slice(yhat);
            xhat = reshape(xhat, x_dim);
            if self.grid.numpy
                xhat = permute(xhat, [2, 1, 3]);
            end
        end
    end
    
end

function [Dn, Dm] = bistochastize(grid, iterations)
    % Compute diagonal matrices to bistochastize a bilateral grid
    if nargin < 2 || isempty(iterations)
        iterations = 10;
    end
    m = grid.splat(ones(grid.npixels, 1));
    n = ones(grid.nvertices, 1);
    for i = 1:iterations
        n = sqrt(n .* m ./ grid.blur(n));
    end
    Dm = sparse(1:grid.nvertices, 1:grid.nvertices, m);
    Dn = sparse(1:grid.nvertices, 1:grid.nvertices, n);
end