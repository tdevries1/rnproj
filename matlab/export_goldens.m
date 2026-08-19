function export_goldens()
% EXPORT_GOLDENS Generate the language-agnostic golden test fixtures for
% the rnproj package from the Matlab reference implementation.
%
% Usage: run this once in Matlab. It addpath's the reference codebase,
% computes every golden case, and writes JSON files into
% <repo>/tests/golden/ (next to this script's parent directory).
%
% All numbers are written with %.17g so they round-trip through IEEE
% doubles exactly. Grids and weighting densities are STORED in the files,
% so cross-language comparisons never depend on optimizer or
% grid-construction drift.

ref_path = '/Users/tjeerddevries/Dropbox/2025 Carr-Madan/Matlab';
assert(isfolder(ref_path), 'Reference Matlab folder not found: %s', ref_path);
addpath(ref_path);

here    = fileparts(mfilename('fullpath'));
out_dir = fullfile(here, '..', 'tests', 'golden');
if ~isfolder(out_dir), mkdir(out_dir); end

meta = struct('language', 'matlab', 'script', 'matlab/export_goldens.m', ...
              'matlab_version', version, 'date', datestr(now, 'yyyy-mm-dd'));

case_bs_dense(out_dir, meta);
case_bs_sparse(out_dir, meta);
case_vg_fixed_params(out_dir, meta);
case_carr_madan(out_dir, meta);
case_cdf_constrained(out_dir, meta);

fprintf('All golden files written to %s\n', out_dir);
end


% ======================================================================= %
function [C, P] = black_prices(F, K, T, r, sigma)
df  = exp(-r * T);
srt = sigma * sqrt(T);
d1  = log(F ./ K) / srt + 0.5 * srt;
d2  = d1 - srt;
C   = df * (F * normcdf(d1) - K .* normcdf(d2));
P   = df * (K .* normcdf(-d2) - F * normcdf(-d1));
end

function w = lognormal_pdf(s, F, sigma, T)
srt = sigma * sqrt(T);
mu  = log(F) - 0.5 * srt^2;
w   = exp(-0.5 * ((log(s) - mu) / srt).^2) ./ (s * srt * sqrt(2*pi));
end


% ======================================================================= %
function case_bs_dense(out_dir, meta)
F = 100; sigma = 0.2; T = 1/12; r = 0.02;
K = linspace(60, 150, 40)';
[C, P] = black_prices(F, K, T, r, sigma);
S_t = F * exp(-r * T);

grid = linspace(40, 220, 2000)';
w    = lognormal_pdf(grid, F, sigma, T);

% projection estimates (generation-3 reference API)
[E_S2, beta_S2] = ols_projection(C, K, P, K, @(x) x.^2,        grid, w, T, r, F);
E_S3            = ols_projection(C, K, P, K, @(x) x.^3,        grid, w, T, r, F);
E_S4            = ols_projection(C, K, P, K, @(x) x.^4,        grid, w, T, r, F);
E_neglogSF      = ols_projection(C, K, P, K, @(x) log(F ./ x), grid, w, T, r, F);
E_SF2           = ols_projection(C, K, P, K, @(x) (x / F).^2,  grid, w, T, r, F);

x_cdf   = linspace(70, 140, 21)';
cdf_raw = ols_projection(C, K, P, K, @(s) double(s <= x_cdf'), grid, w, T, r, F);

pdf_vals = ols_weighted_pricing_pdf(C, K, P, K, grid, S_t, T, 0, r, F, [], w, 0);
idx_pdf  = round(linspace(1, numel(grid), 21))';

j = json_writer(fullfile(out_dir, 'bs_dense.json'));
j.head('bs_dense', meta);
j.obj_open('inputs');
j.num('forward', F);  j.num('maturity', T);  j.num('rate', r);  j.num('sigma', sigma);
j.vec('strikes', K);  j.vec('call_prices', C);  j.vec('put_prices', P);
j.vec('grid', grid);  j.vec('weights', w);
j.obj_close();
j.obj_open('outputs');
j.out('E_S2', E_S2, 1e-10);  j.out('E_S3', E_S3, 1e-10);  j.out('E_S4', E_S4, 1e-10);
j.out('E_neglogSF', E_neglogSF, 1e-10);  j.out('E_SF2', E_SF2, 1e-10);
j.outv('beta_S2', beta_S2, 1e-5);
j.vec('x_cdf', x_cdf);  j.outv('cdf_raw', cdf_raw(:), 1e-8);
j.vec('pdf_index_1based', idx_pdf);  j.outv('pdf_values', pdf_vals(idx_pdf), 1e-8);
j.obj_open('analytic');
j.num('E_S2', F^2 * exp(sigma^2 * T));
j.obj_close();
j.obj_close();
j.close();
end


function case_bs_sparse(out_dir, meta)
% sparse OTC-style chain, no OTM filter (ols_projection_sparse path).
F = 100; sigma = 0.2; T = 1/12; r = 0.02;
K = [85; 93; 100.5; 108; 118];
[C, P] = black_prices(F, K, T, r, sigma);

grid = linspace(70, 140, 1500)';
w    = lognormal_pdf(grid, F, sigma, T);

[E_S2, beta_S2] = ols_projection_sparse(C, K, P, K, @(x) x.^2, grid, w, T, r, F);
E_neglogSF      = ols_projection_sparse(C, K, P, K, @(x) log(F ./ x), grid, w, T, r, F);

j = json_writer(fullfile(out_dir, 'bs_sparse.json'));
j.head('bs_sparse', meta);
j.obj_open('inputs');
j.num('forward', F);  j.num('maturity', T);  j.num('rate', r);  j.num('sigma', sigma);
j.vec('strikes', K);  j.vec('call_prices', C);  j.vec('put_prices', P);
j.vec('grid', grid);  j.vec('weights', w);
j.obj_close();
j.obj_open('outputs');
j.out('E_S2', E_S2, 1e-10);
j.out('E_neglogSF', E_neglogSF, 1e-10);
j.outv('beta_S2', beta_S2, 1e-5);
j.obj_close();
j.close();
end


function case_vg_fixed_params(out_dir, meta)
% VG density + grid rules at FIXED parameters (no optimizer involved).
% NOTE: T/nu = 0.625 < 1, so the Matlab trapezoid subordinator quadrature
% carries spurious mass at the mode; the Python side reproduces this with
% mixture='trapz' (its default 'quantile' quadrature is the corrected one).
sigma = 0.18; nu = 0.4; theta = -0.12;
F = 100; T = 0.25;
KP = [80; 90; 95];  KC = [105; 115; 130];

[grid, pdf, params_full] = vg_grid_from_params(sigma, nu, theta, T, F, KP, KC);
idx = round(linspace(1, numel(grid), 21))';

j = json_writer(fullfile(out_dir, 'vg_fixed_params.json'));
j.head('vg_fixed_params', meta);
j.obj_open('inputs');
j.num('sigma', sigma);  j.num('nu', nu);  j.num('theta', theta);
j.num('forward', F);  j.num('maturity', T);
j.vec('put_strikes', KP);  j.vec('call_strikes', KC);
j.obj_close();
j.obj_open('outputs');
j.out('omega', params_full.omega, 1e-12);
j.out('grid_min', grid(1), 1e-6);
j.out('grid_max', grid(end), 1e-6);
j.out('n_grid', numel(grid), 0);
j.vec('pdf_index_1based', idx);
j.outv('pdf_values', pdf(idx), 1e-6);
j.obj_close();
j.close();
end


function case_carr_madan(out_dir, meta)
F = 100; sigma = 0.2; T = 1/12; r = 0.02;
K = linspace(60, 150, 40)';
[C, P] = black_prices(F, K, T, r, sigma);
S_t = F * exp(-r * T);   % so the hardcoded F_t = exp(r*T)*S_t inside
                         % carr_madan_cdf equals the true forward

cm_S2_trapz = carr_madan_pricing(C, K, P, K, @(x) x.^2, @(x) 2*ones(size(x)), S_t, T, 0, r, F);
cm_S2_simps = carr_madan_pricing(C, K, P, K, @(x) x.^2, @(x) 2*ones(size(x)), S_t, T, 0, r, F, true);
cm_logS     = carr_madan_pricing(C, K, P, K, @(x) log(x), @(x) -1 ./ x.^2, S_t, T, 0, r, F);
cm_S2_sparse = carr_madan_pricing_sparse(C, K, P, K, @(x) x.^2, @(x) 2*ones(size(x)), F, T, 0, r);

% NOTE: the CDF evaluation grid must extend beyond the strike range,
% otherwise the [min(grid); K; max(grid)] interpolation abscissae inside
% carr_madan_cdf are non-monotone and interp1 errors.
x_cdf  = linspace(55, 160, 21)';
cm_cdf = carr_madan_cdf(C, K, P, K, S_t, T, 0, r, x_cdf);

j = json_writer(fullfile(out_dir, 'carr_madan.json'));
j.head('carr_madan', meta);
j.obj_open('inputs');
j.num('forward', F);  j.num('maturity', T);  j.num('rate', r);  j.num('sigma', sigma);
j.vec('strikes', K);  j.vec('call_prices', C);  j.vec('put_prices', P);
j.obj_close();
j.obj_open('outputs');
j.out('cm_S2_trapz', cm_S2_trapz, 1e-12);
% Simpson implementations differ in their odd-interval endpoint handling
% (Matlab File Exchange simps vs scipy.integrate.simpson): compare loosely.
j.out('cm_S2_simpson', cm_S2_simps, 1e-4);
j.out('cm_logS', cm_logS, 1e-12);
j.out('cm_S2_sparse', cm_S2_sparse, 1e-12);
j.vec('x_cdf', x_cdf);
j.outv('cm_cdf', cm_cdf, 1e-12);
j.obj_close();
j.close();
end


function case_cdf_constrained(out_dir, meta)
F = 100; sigma = 0.2; T = 1/12; r = 0.02;
K = [85; 93; 100.5; 108; 118];
[C, P] = black_prices(F, K, T, r, sigma);
S_t = F * exp(-r * T);

grid = linspace(70, 140, 60)';
cdf  = ols_pricing_cdf_discrete(C, K, P, K, grid, S_t, T, 0, r);

j = json_writer(fullfile(out_dir, 'cdf_constrained.json'));
j.head('cdf_constrained', meta);
j.obj_open('inputs');
j.num('forward', F);  j.num('maturity', T);  j.num('rate', r);  j.num('sigma', sigma);
j.vec('strikes', K);  j.vec('call_prices', C);  j.vec('put_prices', P);
j.vec('grid', grid);
j.obj_close();
j.obj_open('outputs');
j.outv('cdf', cdf(:), 1e-6);
j.obj_close();
j.close();
end


% ======================================================================= %
% Minimal JSON writer with %.17g round-trip precision (Matlab's
% jsonencode does not guarantee full double precision).
function j = json_writer(path)
fid = fopen(path, 'w');
assert(fid > 0, 'cannot open %s', path);
state = struct('fid', fid, 'first', true);

    function put(str)
        fprintf(state.fid, '%s', str);
    end
    function comma()
        if state.first, state.first = false; else, put(','); end
        put(sprintf('\n'));
    end
    function head(name, meta)
        put('{'); state.first = false;
        put(sprintf('\n"schema_version": 1,\n'));
        put(sprintf('"case": "%s",\n', name));
        put(sprintf(['"generator": {"language": "%s", "script": "%s", ' ...
                     '"matlab_version": "%s", "date": "%s"}'], ...
                    meta.language, meta.script, strrep(meta.matlab_version, '"', ''''), meta.date));
    end
    function obj_open(name)
        comma(); put(sprintf('"%s": {', name)); state.first = true;
    end
    function obj_close()
        put(sprintf('\n}')); state.first = false;
    end
    function num(name, v)
        comma(); put(sprintf('"%s": %.17g', name, v));
    end
    function vec(name, v)
        comma(); put(sprintf('"%s": [', name));
        put(strjoin(arrayfun(@(x) sprintf('%.17g', x), v(:)', 'UniformOutput', false), ', '));
        put(']');
    end
    function out(name, v, rtol)
        comma();
        put(sprintf('"%s": {"value": %.17g, "rtol": %.3g}', name, v, rtol));
    end
    function outv(name, v, rtol)
        comma(); put(sprintf('"%s": {"value": [', name));
        put(strjoin(arrayfun(@(x) sprintf('%.17g', x), v(:)', 'UniformOutput', false), ', '));
        put(sprintf('], "rtol": %.3g}', rtol));
    end
    function close_()
        put(sprintf('\n}\n')); fclose(state.fid);
        fprintf('wrote %s\n', path);
    end

j = struct('head', @head, 'obj_open', @obj_open, 'obj_close', @obj_close, ...
           'num', @num, 'vec', @vec, 'out', @out, 'outv', @outv, 'close', @close_);
end
