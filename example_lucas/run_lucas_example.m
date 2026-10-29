function run_lucas_example()
% Example and test for negishi_korder.m: two-country Lucas endowment model
% (lucas_kappa.mod). Compares ln kappa from negishi_korder (orders 2 and 4)
% with the exact ln kappa (time-0 budget constraint solved by quadrature).
% Writes lucas_example.csv. Run from this folder:
%   matlab -singleCompThread -batch run_lucas_example
% Dynare (with its MEX files) must be on the MATLAB path; the skewed case needs
% a Dynare version supporting skewed shocks in the k-order solver (see README).
assert(exist('dynare', 'file') == 2, 'run_lucas_example: add Dynare to the MATLAB path');
addpath('..');
global M_ oo_ %#ok<NUSED>

svals = [0.01 0.02 0.05 0.1 0.15 0.2];
skews = [0 0; -0.5 0.2];
hpar = {{'', 'h2', '', 'h4'}, {'', 'h2', 'h3', 'h4'}};   % no sigma^3 variable when shocks are symmetric
out = [];
for sc = 0:1
    dynare('lucas_kappa', ['-DSKEWCASE=' num2str(sc)], 'noclearall', 'nolog');
    pv = @(name) M_.params(strcmp(cellstr(M_.param_names), name));
    ie1 = strcmp(cellstr(M_.exo_names), 'e1'); ie2 = strcmp(cellstr(M_.exo_names), 'e2');
    for s = svals
        M_.Sigma_e(ie1,ie1) = s^2; M_.Sigma_e(ie2,ie2) = (2*s)^2;
        res = negishi_korder({'W'}, hpar{sc+1});
        lk_exact = exact_lk(s, 2*s, skews(sc+1,:), pv('bet'), pv('rho'), pv('theta'), pv('n'));
        out = [out; s skews(sc+1,:) res.h(2:4) max(abs(res.check(:))) res.nsolves ...
               lk_exact res.h(2) res.lk]; %#ok<AGROW>
    end
end
hdr = {'s','skA','skB','h2','h3','h4','max_abs_check','nsolves','lk_exact','lk_order2','lk_order4'};
writetable(array2table(out, 'VariableNames', hdr), 'lucas_example.csv');
fprintf('\n%6s %5s | %13s %13s %13s | %9s | %10s %10s\n', 's', 'skA', 'h2', 'h3', 'h4', ...
        'check', 'err ord 2', 'err ord 4');
for i = 1:size(out,1)
    r = out(i,:);
    fprintf('%6.3f %5.2f | %13.6e %13.6e %13.6e | %9.1e | %10.2e %10.2e\n', r(1), r(2), r(4:6), r(7), ...
            abs(r(10)-r(9)), abs(r(11)-r(9)));
end
end

function lk = exact_lk(sdA, sdB, sk, bet, rho, theta, n)
% Root of the time-0 budget constraint of country A,
%   C0^-rho (C0 - 1) + bet/(1-bet) E[C^-rho (C - X^(1/theta) Y1^(1-1/theta))] = 0,
% with X = (n Y1^((theta-1)/theta) + (1-n) Y2^((theta-1)/theta))^(theta/(theta-1)),
% C = X/(n + (1-n) kappa), log Y_i = e_i, e_0 = 0, by Gauss-Legendre quadrature.
[xA, wA] = sn_nodes(sdA, sk(1)); [xB, wB] = sn_nodes(sdB, sk(2));
[E1, E2] = ndgrid(xA, xB); Wq = wA * wB';
Y1 = exp(E1); Y2 = exp(E2); q = (theta-1)/theta;
X = (n*Y1.^q + (1-n)*Y2.^q).^(1/q);
inc = X.^(1/theta) .* Y1.^(1-1/theta);
F = @(lk) (1/(n+(1-n)*exp(lk)))^(-rho) * (1/(n+(1-n)*exp(lk)) - 1) + bet/(1-bet) * ...
          sum(sum(Wq .* (X/(n+(1-n)*exp(lk))).^(-rho) .* (X/(n+(1-n)*exp(lk)) - inc)));
lk = fzero(F, [-1 1], optimset('TolX', 1e-15));
end

function [x, wt] = sn_nodes(sd, sk)
% Nodes and weights for expectations under a centred skew normal with
% standard deviation sd and skewness sk (Gauss-Legendre on +-14 sd).
m = 400;
[t, wt] = gauss_legendre(m);
x = 14*sd*t; wt = 14*sd*wt;
if sk == 0
    omega = sd; alpha = 0;
else
    a23 = abs(sk)^(2/3);
    delta = sqrt(pi/2*a23/(a23 + ((4-pi)/2)^(2/3)));
    alpha = sign(sk)*delta/sqrt(1-delta^2);
    omega = sd/sqrt(1-2/pi*delta^2);
end
mu = -omega*alpha/sqrt(1+alpha^2)*sqrt(2/pi);
pdf = 2/omega*normpdf((x-mu)/omega).*normcdf(alpha*(x-mu)/omega);
wt = wt .* pdf;
end

function [t, w] = gauss_legendre(m)
% Golub-Welsch nodes and weights on [-1, 1].
k = 1:m-1;
J = diag(k ./ sqrt(4*k.^2 - 1), 1); J = J + J';
[V, D] = eig(J);
[t, i] = sort(diag(D)); w = 2*V(1,i)'.^2;
end
