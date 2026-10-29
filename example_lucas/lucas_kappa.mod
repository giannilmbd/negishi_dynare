/*
 * Two-country Lucas endowment model with differentiated goods, PPP, iid
 * log endowment shocks e1 (country A) and e2 (country B), complete markets.
 *
 * ln kappa = c2 - c1 is a constant set through auxiliary variables that are
 * exact powers of the perturbation parameter sigma:
 *   a = E_t zz(+1)^2 = sigma^2   (z ~ N(0,1), enters nowhere else)
 *   b = E_t ww(+1)^3 = sw*sigma^3 (w unit-variance skew normal, skewness sw)
 *   lk = h2*a + h3*b/sw + h4*a^2 = h2*sigma^2 + h3*sigma^3 + h4*sigma^4.
 * W is the time-t value, in units of A's marginal utility, of A's future net
 * purchases; the time-0 budget constraint is W_0 = 0 with e_0 = 0.
 *
 * junk is an inert backward-looking variable: Dynare does not solve purely
 * forward models at order > 1.
 *
 * Usage: dynare lucas_kappa -DSKEWCASE=0 (Gaussian) or -DSKEWCASE=1
 */
@#ifndef SKEWCASE
@#define SKEWCASE = 0
@#endif

var c1 c2 p1 p2 W lk a zz junk
@#if SKEWCASE
  b ww
@#endif
;
varexo e1 e2 z
@#if SKEWCASE
  w
@#endif
;
parameters bet rho theta n h2 h3 h4 sw;
bet = 0.99; rho = 2; theta = 3; n = 0.5;
h2 = 0; h3 = 0; h4 = 0; sw = 0.5;

model;
exp(e1) = exp(-theta*p1)*(n*exp(c1) + (1-n)*exp(c2));
n*(exp(p1+e1) - exp(c1)) + (1-n)*(exp(p2+e2) - exp(c2)) = 0;
n*exp((1-theta)*p1) + (1-n)*exp((1-theta)*p2) = 1;
c2 - c1 = lk;
zz = z;
a = zz(+1)^2;
@#if SKEWCASE
ww = w;
b = ww(+1)^3;
lk = h2*a + h3*b/sw + h4*a^2;
@#else
lk = h2*a + h4*a^2;
@#endif
W = exp(-rho*c1)*(exp(c1) - exp(p1+e1)) + bet*W(+1);
junk = 0.9*junk(-1);
end;

steady_state_model;
c1 = 0; c2 = 0; p1 = 0; p2 = 0; W = 0; lk = 0; a = 0; zz = 0; junk = 0;
@#if SKEWCASE
b = 0; ww = 0;
@#endif
end;

shocks;
var e1 = 0.01^2;
var e2 = 0.02^2;
var z = 1;
@#if SKEWCASE
var w = 1;
skew w = 0.5;
skew e1 = -0.5;
skew e2 = 0.2;
@#endif
end;

steady;
stoch_simul(order=4, pruning, k_order_solver, irf=0, periods=0, noprint);
