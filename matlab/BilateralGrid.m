classdef BilateralGrid
    %BILATERALGRID Bilateral grid use for filtering
    %
    %SEE
    %   https://github.com/spillai/bilateral_solver
    
    properties
        sigma_sp    = 32
        sigma_lum   = 8
        sigma_chr   = 8
        npixels     = 0
        dim         = 0
        hash_vec    = []
        nvertices   = 0
        S           = []
        blurs       = {}
        max_val     = 255.0
    end
    
    methods
        function self = BilateralGrid(img, sig_sp, sig_l, sig_ch)
            % convert rgb to special ycbcr using full channel ranges
            img = rgb2ycbcr_full(img);
            if nargin > 1 && ~isempty(sig_sp)
                self.sigma_sp = sig_sp;
            end
            if nargin > 2 && ~isempty(sig_l)
                self.sigma_lum = sig_l;
            end
            if nargin > 3 && ~isempty(sig_ch)
                self.sigma_chr = sig_ch;
            end
            [h, w, ~]   = size(img);
            [Iy, Ix]    = meshgrid(1:w, 1:h);
            x_coords    = round(Ix / self.sigma_sp);
            y_coords    = round(Iy / self.sigma_sp);
            l_coords    = round(img(:, :, 1) / self.sigma_lum);
            ch_coords   = round(img(:, :, 2:3) / self.sigma_chr);
            coords  = cat(3, x_coords, y_coords, l_coords, ch_coords);
            coords_flat = reshape(coords, [h * w, 5]);
            self.npixels = h * w;
            self.dim     = 5;
            self.hash_vec = (self.max_val .^ (0:4)); % + 1?
            % construct S and B matrix
            self = self.compute_factorization(coords_flat);
        end
        
        
        function self = compute_factorization(self, coords_flat)
            % Hash each coordinate in grid to a unique value
            hashed_coords = self.hash_coords(coords_flat);
            [unique_hashes, unique_idx, idx] = unique(hashed_coords); 
            % Identify unique set of vertices
            unique_coords = coords_flat(unique_idx, :);
            self.nvertices = size(unique_coords, 1);
            % Construct sparse splat matrix that maps from pixels to vertices
            % self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
            self.S = sparse(idx, 1:self.npixels, ones(self.npixels, 1));
            % Construct sparse blur matrices.
            % Note that these represent [1 0 1] blurs, excluding the central element
            self.blurs = cell(1, self.dim);
            for d = 1:self.dim
                blur = sparse(self.nvertices, self.nvertices);
                for offset = [-1, 1]
                    offset_vec = zeros(1, self.dim);
                    offset_vec(:, d) = offset;
                    neighbor_coords = bsxfun(@plus, unique_coords, (offset_vec));
                    neighbor_hash = self.hash_coords(neighbor_coords);
                    % [valid_coord, idx] = get_valid_idx(unique_hashes, neighbor_hash);
                    [valid, idx] = ismember(neighbor_hash, unique_hashes);
                    valid_coord = find(valid);
                    idx = idx(valid);
                    blur = blur + sparse(valid_coord, idx, ones(numel(idx), 1), ...
                                            self.nvertices, self.nvertices);
                        %csr_matrix((np.ones((len(valid_coord),)),
                        %                      (valid_coord, idx)),
                        %                     shape=(self.nvertices, self.nvertices))
                end
                self.blurs{d} = blur;
            end
        end
        
        function h = hash_coords(self, coord) % coord is Nx5, h is Nx1
            coord = reshape(coord, [], self.dim);
            h = sum(bsxfun(@times, coord, self.hash_vec), 2);
        end
        
        function y = splat(self, x)
            assert(size(x, 1) == self.npixels, 'Invalid splat size: %d <> %d', size(x, 1), self.npixels);
            y = zeros(size(self.S, 1), size(x, 2));
            for i = 1:size(x, 2)
                y(:, i) = self.S * x(:, i);
            end
        end
        
        function x = slice(self, y)
            assert(size(y, 1) == self.nvertices, 'Invalid slice size: %d <> %d', size(y, 1), self.nvertices);
            x = zeros(size(self.S, 2), size(y, 2));
            for i = 1:size(y, 2)
                x(:, i) = self.S' * y(:, i);
            end
        end
        
        function b = blur(self, x)
            % blur a bilateral-space vector with a [1 2 1] kernel in each
            % dimension
            assert(size(x, 1) == self.nvertices, 'Invalid vector size');
            b = 2 * self.dim * x;
            for i = 1:self.dim
                b = b + self.blurs{i} * x;
            end
        end
        
        function z = filter(self, x)
            if ~isfloat(x)
                x = im2double(x);
            end
            x_dim = size(x);
            x = reshape(x, self.npixels, []);
            % apply bilateral filter to an input x
            z = self.slice(self.blur(self.splat(x))) ./ ...
                self.slice(self.blur(self.splat(ones(size(x)))));
            z = reshape(z, x_dim);
        end
    end
    
end

function conv = rgb2ycbcr_full( img )
    if isfloat(img)
        img = img * 255.0;
    else
        img = double(img);
    end
    A = [ 0.299, 0.587, 0.114; ...
         -0.168736, -0.331264,  0.5; ...
         0.5, -0.418688, -0.081312 ...
    ];
    b = [0, 128, 128];
    conv = reshape(img, [], 3) * A;
    conv = bsxfun(@plus, conv, reshape(b, 1, []));
    conv = reshape(conv, size(img));
end
