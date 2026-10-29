# Negishi weights to order K in complete-markets Dynare models

`negishi_korder.m` computes the risk-sharing constants (Negishi weights) of a
complete-markets model to any perturbation order K >= 2, using Dynare's k-order
solver. It generalises the second-order procedure of the Online Appendix of
Corsetti, Lipińska and Lombardo, "International Risk Sharing and Wealth
Allocation with Higher Order Cumulants" (Online Appendix, section "A
useful efficient procedure", the `dr.G` / `ghs2` swap) to higher orders, without modifying Dynare.

## Problem

Under complete markets, the ratio of marginal utilities of the numeraire across
countries is constant over time:

    lam_i,t = kappa_i * lam_1,t ,   i = 2..N.

The constants kappa_i are not determined by the dynamic equations. They are set by
the time-0 budget constraints (present value of net imports = initial assets),
which depend on risk. Write ln kappa_i = ln kappa_i^(0) + sum_{j>=2} h_{i,j} sigma^j,
with kappa_i^(0) the deterministic-steady-state value and sigma the perturbation
parameter that scales all shocks (sigma = 1 in Dynare).

## Why each order is a linear problem

Let Z_i,0 be the time-0 budget-constraint residual of country i, evaluated with
the endogenous states at the deterministic steady state and zero shocks at t = 0.
Its sigma^j coefficient is

    z_j = J h_j + N_j(h_2, ..., h_{j-1}),     J = dZ_0 / d ln kappa  (deterministic SS).

ln kappa is time-invariant and enters the equations like a parameter, so by Faà
di Bruno's formula h_j appears at order j only through the first derivative J.
Every other order-j term involves lower-order h's. Setting z_j = 0 gives

    h_2 = -J^{-1} z_2(h = 0),   h_j = -J^{-1} z_j(h_2..h_{j-1}, h_j = 0),  j = 3..K.

The nonlinear part N_j is computed by Dynare: a solve with the lower-order h's in
place returns z_j directly.

## Setting up a model

1. **Risk sharing.** For i = 2..N:

       lam_i = lam_1*exp(lkbar_i + lk_i);

   where `lkbar_i` is a parameter equal to log(lam_i/lam_1) at the deterministic
   steady state (set in the steady-state file), and `lk_i` is a variable.

2. **Powers of sigma.** Add auxiliary shocks that enter nowhere else and variables
   whose decision rules are exact powers of sigma:

       zz = z;  a = zz(+1)^2;      // z ~ N(0,1):           a  = sigma^2
       ww = w;  s3 = ww(+1)^3;     // w unit variance, skewness sw:  s3 = sw*sigma^3
       lk_i = h2_i*a + h3_i*s3/sw + h4_i*a^2;

   `h2_i, h3_i, h4_i` are parameters (initialised to 0). Higher orders: sigma^5 =
   a*s3/sw, sigma^6 = a^3, and so on. The sigma^3 term is needed only when some
   shock is skewed (otherwise all odd-order residuals are zero; omit `s3`, `ww`,
   `w`). A skewed auxiliary shock needs the Dynare fork with skewed shocks
   (`skew w = 0.5;` in the shocks block). At the deterministic steady state a = 0, so
   lk_i = 0.

3. **Budget constraints in present value.** Replace each country's flow budget
   constraint (and any bond-market clearing, debt-elastic premium, etc.) by

       W_i = lam_i*NX_i + beta*SDFgrowth(+1)*W_i(+1);
       Z_i = W_i + lam_i*(gross initial assets)_i;

   with NX_i net exports in numeraire units and SDFgrowth the detrending factor of
   the discount (e.g. Gam(+1)^(1-rho) with a stochastic trend). Z_i,0 = 0 is the
   time-0 budget constraint. Impose all goods-market clearing conditions
   explicitly (no market is dropped by Walras' law any more). Only Z_2..Z_N are
   conditions; Z_1 = 0 then follows from aggregation and serves as a check.

4. **Solve.** In the .mod file: `stoch_simul(order=K, pruning, k_order_solver, ...);`
   Then in MATLAB:

       res = negishi_korder({'Z_2','Z_3'}, ...
                            {'', 'h2_2', 'h3_2', 'h4_2';
                             '', 'h2_3', 'h3_3', 'h4_3'});

   Pass '' for a missing sigma^j parameter (e.g. column 3 with symmetric shocks).
   On exit `M_.params` holds the h's and `oo_.dr` is the solution at the computed
   weights. `res.check` holds the sigma^j terms of Z at the solution (should be at
   rounding level). The routine uses m + K model solutions (m = number of weights).

The pure sigma^j terms are read from the last column of `oo_.dr.pruning.g_j`
(the 1/j! factor is included).

## Requirements and notes

- Dynare's k-order solver with pruning (`oo_.dr.pruning`). Skewed shocks (and the
  sigma^3 auxiliary variable) need a Dynare version whose k-order solver uses
  skewed-shock moments, e.g. the fork https://github.com/giannilmbd/dynare, branch
  `skewed-shocks-third-moments`.
- Dynare does not solve purely forward-looking models at order > 1. A model without
  backward-looking variables needs an inert one, e.g. `junk = 0.9*junk(-1);`.
- J must be invertible (the routine stops if rcond(J) < 1e-12).
- The time-0 convention (states at the deterministic steady state, zero shocks at
  t = 0) is the one of the second-order procedure in the Online Appendix.

## Example and validation

`example_lucas/`: two-country Lucas endowment model with differentiated goods
(PPP, n = 1/2, beta = 0.99, rho = 2, theta = 3, iid log endowment shocks with
standard deviations sd_A = s, sd_B = 2s; Gaussian case and skew-normal case with
skewness -0.5 and 0.2). The exact ln kappa solves the time-0 budget constraint by
Gauss-Legendre quadrature over the shock densities.

Run in `example_lucas/` with Dynare on the MATLAB path:

    matlab -singleCompThread -batch run_lucas_example

Output: `lucas_example.csv` and the table printed in `run_lucas_example.log`.

Results:
- the time-0 budget-constraint residuals at the computed weights are below 1e-15;
- error of ln kappa against the exact solution, Gaussian shocks: O(s^4) at order 2,
  O(s^6) at order 4 (s = 0.2: 2.0e-3 and 1.6e-4); skewed shocks: O(s^5) at order 4
  (s = 0.2: 1.1e-4);
- h_2, h_3, h_4 coincide with an independent symbolic perturbation of the same model
  to relative 1e-14.

## Licence

GNU General Public License v3.0 (see LICENSE), as Dynare.
