function res = negishi_korder(cond_vars, h_params, tol)
% NEGISHI_KORDER  Negishi weights (risk-sharing constants) to order K in a
% complete-markets Dynare model, by the auxiliary-variable recursion of README.md.
%
%   res = negishi_korder(cond_vars, h_params)
%   res = negishi_korder(cond_vars, h_params, tol)
%
% Call after `dynare <model>` has run stoch_simul(order=K, pruning, k_order_solver)
% (globals M_, options_, oo_).
%
% cond_vars : cellstr, m endogenous variables Z_1..Z_m whose value at t = 0
%             (states at the deterministic steady state, zero shocks) must be 0:
%             the time-0 budget constraints, one per free Negishi weight.
% h_params  : m x K cellstr; h_params{i,j} is the parameter multiplying sigma^j in
%             ln kappa_i (column 1 unused). Empty where no such parameter exists;
%             then the sigma^j residual must already be zero (e.g. odd j with
%             symmetric shocks), otherwise an error is raised.
% tol       : absolute tolerance on residuals for the empty-parameter case (1e-12).
%
% res.h      m x K values of the h parameters (sigma^j coefficients, 1/j! included),
%            also stored in M_.params
% res.lk     m x 1 sum over j of res.h (ln kappa_i - ln kappa_i at the deterministic
%            steady state, at sigma = 1)
% res.J      m x m Jacobian dZ_0/d ln kappa at the deterministic steady state
% res.r0     m x K pure sigma^j terms of Z_0 with all h = 0
% res.check  m x K pure sigma^j terms of Z_0 at the solution (should be ~0)
% res.nsolves number of model solutions used
%
% On exit oo_.dr is the solution at the computed Negishi weights.

global M_ options_ oo_

if nargin < 3, tol = 1e-12; end
K = options_.order;
assert(K >= 2, 'negishi_korder: needs order >= 2');
assert(isfield(oo_.dr, 'pruning'), ...
    'negishi_korder: run stoch_simul with pruning and k_order_solver (oo_.dr.pruning missing)');
m = numel(cond_vars);
assert(isequal(size(h_params), [m K]), 'negishi_korder: h_params must be %d x %d', m, K);

pn = cellstr(M_.param_names);
en = cellstr(M_.endo_names);
iz = zeros(m, 1);
for i = 1:m
    k = find(strcmp(en, cond_vars{i}));
    assert(numel(k) == 1, 'negishi_korder: variable %s not found', cond_vars{i});
    iz(i) = find(oo_.dr.order_var == k);
end
ip = zeros(m, K);
for i = 1:m
    for j = 2:K
        if ~isempty(h_params{i,j})
            k = find(strcmp(pn, h_params{i,j}));
            assert(numel(k) == 1, 'negishi_korder: parameter %s not found', h_params{i,j});
            ip(i,j) = k;
        end
    end
end
assert(all(ip(:,2) > 0), 'negishi_korder: every weight needs a sigma^2 parameter');

nsolves = 0;
h = zeros(m, K);
setall(h);
r0 = pure_sigma();

% Jacobian from the sigma^2 terms, which are exactly linear in the h_2's
J = zeros(m);
for i = 1:m
    hi = zeros(m, K); hi(i,2) = 1;
    setall(hi);
    r = pure_sigma();
    J(:,i) = r(:,2) - r0(:,2);
end
assert(rcond(J) > 1e-12, 'negishi_korder: Jacobian dZ/dln(kappa) is singular (rcond %g)', rcond(J));

h(:,2) = -J \ r0(:,2);
for j = 3:K
    setall(h);
    r = pure_sigma();
    if all(ip(:,j) > 0)
        h(:,j) = -J \ r(:,j);
    elseif any(ip(:,j) > 0)
        error('negishi_korder: sigma^%d parameters given for some weights only', j);
    elseif max(abs(r(:,j))) > tol
        error(['negishi_korder: sigma^%d residual %g but no sigma^%d parameter ' ...
               '(add an auxiliary variable equal to sigma^%d, see README)'], j, max(abs(r(:,j))), j, j);
    end
end
setall(h);
check = pure_sigma();

res = struct('h', h, 'lk', sum(h, 2), 'J', J, 'r0', r0, 'check', check, 'nsolves', nsolves);

    function setall(hh)
        for ii = 1:m
            for jj = 2:K
                if ip(ii,jj) > 0, M_.params(ip(ii,jj)) = hh(ii,jj); end
            end
        end
    end

    function r = pure_sigma()
        [info, oo_] = stoch_simul(M_, options_, oo_, {});
        nsolves = nsolves + 1;
        if info(1), error('negishi_korder: stoch_simul failed, info = %d', info(1)); end
        r = zeros(m, K);
        for jj = 1:K
            g = oo_.dr.pruning.(sprintf('g_%d', jj));
            r(:,jj) = g(iz, end);
        end
    end
end
