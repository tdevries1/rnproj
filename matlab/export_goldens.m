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
case_fx_quotes(out_dir, meta);
case_fx_bivariate(out_dir, meta);

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


function case_fx_quotes(out_dir, meta)
% (ATM, RR, BF) quotes -> strikes and GK prices. Formulas transcribed from
% fx_make_prices.m local functions (they are private subfunctions there).
atm = 0.10; rr25 = -0.015; bf25 = 0.003; rr10 = -0.025; bf10 = 0.008;
F = 1.10; T = 1/12; Dd = 0.998; Df = 0.997;

C25 = atm + 0.5*(bf25 + rr25);  P25 = atm + 0.5*(bf25 - rr25);
C10 = atm + 0.5*(bf10 + rr10);  P10 = atm + 0.5*(bf10 - rr10);

K_ATM = F * exp(0.5 * atm^2 * T);
K_C25 = strike_from_spot_delta_(0.25, true,  F, C25, T, Df);
K_P25 = strike_from_spot_delta_(0.25, false, F, P25, T, Df);
K_C10 = strike_from_spot_delta_(0.10, true,  F, C10, T, Df);
K_P10 = strike_from_spot_delta_(0.10, false, F, P10, T, Df);

price_ATM_put = gk_put_(F, K_ATM, atm, T, Dd);
price_P25     = gk_put_(F, K_P25, P25, T, Dd);
price_P10     = gk_put_(F, K_P10, P10, T, Dd);
price_C25     = gk_call_(F, K_C25, C25, T, Dd);
price_C10     = gk_call_(F, K_C10, C10, T, Dd);

j = json_writer(fullfile(out_dir, 'fx_quotes.json'));
j.head('fx_quotes', meta);
j.obj_open('inputs');
j.num('atm', atm);  j.num('rr25', rr25);  j.num('bf25', bf25);
j.num('rr10', rr10);  j.num('bf10', bf10);
j.num('forward', F);  j.num('maturity', T);
j.num('domestic_df', Dd);  j.num('foreign_df', Df);
j.obj_close();
j.obj_open('outputs');
j.outv('put_strikes',  [K_P10; K_P25; K_ATM], 1e-12);
j.outv('call_strikes', [K_C25; K_C10], 1e-12);
j.outv('put_prices',   [price_P10; price_P25; price_ATM_put], 1e-12);
j.outv('call_prices',  [price_C25; price_C10], 1e-12);
j.obj_close();
j.close();
end


function case_fx_bivariate(out_dir, meta)
% Synthetic FX triangle under a joint-lognormal Q with rho = 0.6, priced
% analytically (Black legs + Margrabe cross), run through the reference
% joint-projection code path of cm_simulation8.m (basis, stacked prices,
% covariance, tail, marginals, Hoeffding cells).
rho = 0.6; F1 = 1.10; F2 = 1.30; sig1 = 0.10; sig2 = 0.08; T = 1/12;
n_grid = 150; a1 = 0.97; a2 = 0.97;

z = [-1.6; -0.8; 0; 0.8; 1.6];
K1 = F1 * exp(z * sig1 * sqrt(T));
K2 = F2 * exp(z * sig2 * sqrt(T));
[C1, P1] = black_prices(F1, K1, T, 0, sig1);
[C2, P2] = black_prices(F2, K2, T, 0, sig2);

f3 = F1 / F2;
sx = sqrt(sig1^2 + sig2^2 - 2*rho*sig1*sig2);
K3 = f3 * exp(z * max(sx, 0.02) * sqrt(T));
srt = sx * sqrt(T);
d1 = (log(F1 ./ (K3 * F2)) + 0.5 * srt^2) / srt;
C3 = (F1 * normcdf(d1) - K3 * F2 .* normcdf(d1 - srt)) / F2;  % Rf3 = 1
P3 = C3 - (F1 - K3 * F2) / F2;                                % parity

% puts = lower 3 strikes, calls = upper 2 (as in the OTC 5-strike layout)
Kp1 = K1(1:3); Kc1 = K1(4:5);  Pp1 = P1(1:3); Cc1 = C1(4:5);
Kp2 = K2(1:3); Kc2 = K2(4:5);  Pp2 = P2(1:3); Cc2 = C2(4:5);
Kp3 = K3(1:3); Kc3 = K3(4:5);  Pp3 = P3(1:3); Cc3 = C3(4:5);

x1 = linspace(0.95*min(K1), 1.02*max(K1), n_grid)';
x2 = linspace(0.95*min(K2), 1.02*max(K2), n_grid)';
[x1_ten, x2_ten] = meshgrid(x1, x2);
x1_ten = x1_ten(:);  x2_ten = x2_ten(:);

% reference basis and stacked prices (r = 0, all gross rates 1)
Phi = [ones(size(x1_ten)), x1_ten, max(Kp1' - x1_ten, 0), max(x1_ten - Kc1', 0), ...
       x2_ten, max(Kp2' - x2_ten, 0), max(x2_ten - Kc2', 0), ...
       x2_ten .* max(Kp3' - x1_ten ./ x2_ten, 0), ...
       x2_ten .* max(x1_ten ./ x2_ten - Kc3', 0)];
prices_stacked = [1; F1; Pp1; Cc1; F2; Pp2; Cc2; F2 * [Pp3; Cc3]];

target  = (x1_ten - F1) .* (x2_ten - F2);
cov_hat = (Phi \ target)' * prices_stacked;

phi_marginal = @(x, Kp, Kc) [ones(numel(x),1), x, max(Kp' - x, 0), max(x - Kc', 0)];
var1 = (phi_marginal(x1, Kp1, Kc1) \ (x1 - F1).^2)' * [1; F1; Pp1; Cc1];
var2 = (phi_marginal(x2, Kp2, Kc2) \ (x2 - F2).^2)' * [1; F2; Pp2; Cc2];
corr_hat = cov_hat / sqrt(var1 * var2);

target_tail_fun = @(u1,u2,Fa,Fb,aa,bb) double((u1./Fa <= aa) & (u2./Fb <= bb));
tail_risk = (Phi \ target_tail_fun(x1_ten, x2_ten, F1, F2, a1, a2))' * prices_stacked;

beta_hat_marg = @(x, Kp, Kc, f, a) phi_marginal(x, Kp, Kc) \ (x./f <= a);
marg1 = beta_hat_marg(x1, Kp1, Kc1, F1, a1)' * [1; F1; Pp1; Cc1];
marg2 = beta_hat_marg(x2, Kp2, Kc2, F2, a2)' * [1; F2; Pp2; Cc2];

x1_coarse = linspace(min(x1), max(x1), 4)' / F1;
x2_coarse = linspace(min(x2), max(x2), 4)' / F2;
hoef = hoeffdingCovCells(x1_coarse, x2_coarse, x1, x2, x1_ten, x2_ten, Phi, ...
                         [F1; Pp1; Cc1], [F2; Pp2; Cc2], F2 * [Pp3; Cc3], ...
                         F1, F2, Kp1, Kc1, Kp2, Kc2, ...
                         target_tail_fun, beta_hat_marg);

j = json_writer(fullfile(out_dir, 'fx_bivariate.json'));
j.head('fx_bivariate', meta);
j.obj_open('inputs');
j.num('rho', rho);  j.num('F1', F1);  j.num('F2', F2);
j.num('sigma1', sig1);  j.num('sigma2', sig2);  j.num('maturity', T);
j.num('n_grid', n_grid);  j.num('a1', a1);  j.num('a2', a2);
j.vec('K1', K1);  j.vec('C1', C1);  j.vec('P1', P1);
j.vec('K2', K2);  j.vec('C2', C2);  j.vec('P2', P2);
j.vec('K3', K3);  j.vec('C3', C3);  j.vec('P3', P3);
j.obj_close();
j.obj_open('outputs');
j.out('cov_hat', cov_hat, 1e-8);
j.out('corr_hat', corr_hat, 1e-8);
j.out('var1', var1, 1e-8);
j.out('var2', var2, 1e-8);
j.out('tail_risk', tail_risk, 1e-6);
j.out('marginal1', marg1, 1e-8);
j.out('marginal2', marg2, 1e-8);
j.out('hoeffding_total', hoef.Ctotal, 1e-6);
j.outv('hoeffding_cells', hoef.Ccell(:), 1e-6);
j.obj_close();
j.close();
end


function c = gk_call_(F, K, sigma, T, Dd)
d1 = (log(F./K) + 0.5*(sigma.^2).*T) ./ (sigma.*sqrt(T));
c  = Dd .* (F.*normcdf(d1) - K.*normcdf(d1 - sigma.*sqrt(T)));
end

function p = gk_put_(F, K, sigma, T, Dd)
d1 = (log(F./K) + 0.5*(sigma.^2).*T) ./ (sigma.*sqrt(T));
p  = Dd .* (K.*normcdf(-(d1 - sigma.*sqrt(T))) - F.*normcdf(-d1));
end

function K = strike_from_spot_delta_(deltaAbs, isCall, F, sigma, T, Df)
if isCall
    DeltaF = min(max(deltaAbs ./ Df, 1e-6), 1-1e-6);
else
    DeltaF = min(max(1 - (deltaAbs ./ Df), 1e-6), 1-1e-6);
end
d1 = norminv(DeltaF);
K  = F .* exp(-sigma.*sqrt(T).*d1 + 0.5*(sigma.^2).*T);
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
