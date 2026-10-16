import argparse
import math
import numpy as np
from mpmath import mp
from scipy.optimize import brentq
from scipy.special import gammaln, lambertw, logsumexp
from rcg import bell_generalized_list

# Numerical checks of the asymptotic mean and variance of the number of cliques C, the number of edges M and
# the degree D of a uniformly chosen vertex of the random cluster graph CG_{n,p}, for p=1/2 (critical) and
# p=1/4 (subcritical). For each regime, we compare
#   1. the exact moments with the saddle-point moments for n <= 1000, and
#   2. the saddle-point moments with the asymptotes from the paper for much larger n.
# Run `python verification.py`, or `python verification.py --quick` for a faster run.

STATS = ('C', 'M', 'D')


# Exact means and variances of C, M and D for CG_{i,p} for all i<=n, using the generalised Bell numbers
# from rcg.py. The clique of vertex i has size S_i with law (Corollary 2.5)
#   P(S_i=s) = {i-1 \choose s-1} w^{s \choose 2} B_{i-s}(w) / B_i(w),
# and the remaining vertices form a CG_{i-s,p}. Hence C_i = 1 + C_{i-S}, M_i = {S \choose 2} + M_{i-S} and
# D_i = S_i - 1. Variances follow from the law of total variance, which avoids catastrophic cancellation.
# Returns {i: {'C': (mean, var), 'M': (mean, var), 'D': (mean, var)}}.
def moments_mp(n, p, bells=None):
    if bells is None:
        bells = bell_generalized_list(n, p=p)
    alpha = mp.log(p) - mp.log(1-p)
    contribution = {'C': lambda s: 1, 'M': lambda s: math.comb(s, 2)}
    means = {X: [mp.mpf(0)] for X in contribution}
    variances = {X: [mp.mpf(0)] for X in contribution}
    result = {}
    for i in range(1, n+1):
        sizes = range(1, i+1)
        probs = [math.comb(i-1, s-1) * mp.exp(alpha*math.comb(s, 2)) * bells[i-s] / bells[i] for s in sizes]
        result[i] = {}
        for X, f in contribution.items():
            xs = [f(s) + means[X][i-s] for s in sizes]
            mean = mp.fsum(P*x for P, x in zip(probs, xs))
            var = mp.fsum(P*(variances[X][i-s] + (x-mean)**2) for s, P, x in zip(sizes, probs, xs))
            means[X].append(mean)
            variances[X].append(var)
            result[i][X] = (mean, var)
        mean = mp.fsum(P*(s-1) for s, P in zip(sizes, probs))
        result[i]['D'] = (mean, mp.fsum(P*(s-1-mean)**2 for s, P in zip(sizes, probs)))
    return result


# ----------------------------------------------------------------------------------------------------------
# Saddle-point moments
# ----------------------------------------------------------------------------------------------------------

# log of the terms w^{j \choose 2} x^j / j! of C_0(w,x), with a = log(1/w) and x = tau (1/w)^(tau-1/2).
def _log_C_terms(tau, a):
    j = np.arange(1, int(tau + 30*math.sqrt(tau)) + 100, dtype=float)
    log_x = math.log(tau) + (tau-0.5)*a
    return j, -a*j*(j-1)/2 + j*log_x - gammaln(j+1)


# tau_n: the solution of C_1(w, tau (1/w)^(tau-1/2)) = n (Lemma B.3). Equivalently, tau = W(a x/sqrt(w))/a
# with W the Lambert W-function and x the saddle point C_1(w,x) = n (footnote 5 of Lemma B.1).
# For p=1/2 (w=1), C_1(1,x) = x e^x, so that tau = x = W(n).
def tau_n(n, p):
    a = math.log((1-p)/p)
    if a == 0:
        return float(lambertw(n).real)

    def f(tau):
        j, log_terms = _log_C_terms(tau, a)
        return logsumexp(log_terms + np.log(j)) - math.log(n)
    return brentq(f, 1e-3, 100, xtol=1e-14)


# The means and variances that appear in the proofs of Propositions 5.2 and 5.4, before the periodic
# simplifications: mu = H'(0) and sigma^2 = H''(0), evaluated at the saddle point x = e^{gamma(0)} with
# C_1(w,x) = n. With q_j = w^{j \choose 2} x^j/j! (so that C_r = sum_j j^r q_j):
#   C:  mu = C_0,               sigma^2 = C_0 - C_1^2/C_2                 (proof of Prop. 5.2)
#   M:  mu = (C_2 - C_1)/2,     sigma^2 = (C_2 C_4 - C_3^2)/(4 C_2)       (eq. (5.22)-(5.23))
# For D we use the analogous saddle-point approximation of Corollary 2.5: replacing B_{n-s}(w)/B_n(w) by
# (n-s)!/n! x^s gives P(S_n=s) ≈ s q_s/C_1, i.e. S_n is approximately the size-biased version of q, so
#   D:  mu = C_2/C_1 - 1,       sigma^2 = C_3/C_1 - (C_2/C_1)^2.
# For p=1/2 (w=1, x=W(n)) these are the moments of the Poissonised partition of Section 4 with
# lambda = W(n), and evaluate to n/W - 1, n/(W(1+W)) - 1, nW/2, nW(2+2W+W^2)/(4(1+W)), W and W.
# The variances are evaluated as variances of j under normalised weights to avoid cancellation.
def saddle_point_moments(n, p):
    a = math.log((1-p)/p)
    j, log_terms = _log_C_terms(tau_n(n, p), a)
    q = np.exp(log_terms - log_terms.max())
    q /= q.sum()
    Ej = q @ j
    Ej2 = q @ j**2
    Z = n/Ej  # = C_0, since C_1 = Z E[j] = n
    var_q = q @ (j-Ej)**2
    size_biased = q*j / Ej  # weights proportional to j q_j
    mean_size_biased = size_biased @ j
    pi = q*j**2 / Ej2  # weights proportional to j^2 q_j
    var_pi = pi @ (j - pi @ j)**2
    return {
        'C': (Z, Z*var_q/Ej2),
        'M': (Z*(Ej2-Ej)/2, Z*Ej2*var_pi/4),
        'D': (mean_size_biased - 1, size_biased @ (j-mean_size_biased)**2),
    }


# ----------------------------------------------------------------------------------------------------------
# Asymptotes
# ----------------------------------------------------------------------------------------------------------

# The periodic functions e_{w,l}(tau) = sum_{t+tau in Z} t^l w^{t^2/2} from Lemma B.2.
def periodic_e(tau, p, l):
    a = math.log((1-p)/p)
    t = np.arange(-300, 301) + (math.ceil(tau) - tau)
    return np.sum(t**l * np.exp(-a*t**2/2))


# The asymptotic means and variances from the paper, and their names.
# Critical regime p=1/2, with W the Lambert W-function:
#   E[C] ~ n/W(n),    Var(C) ~ n/W(n)^2       (Theorem 1.1 (ii); eqs. (4.12)-(4.13))
#   E[M] ~ n W(n)/2,  Var(M) ~ n W(n)^2/4     (Theorem 1.3 (ii); Lemmas 4.1 and 4.8)
#   E[D] ~ W(n),      Var(D) ~ W(n)           (Theorem 1.4 (ii): D is asymptotically Poisson with mean W(n))
# Subcritical regime p<1/2, with tau = tau_n from Lemma B.3 and Var(T) = (e_0 e_2 - e_1^2)/e_0^2 (Remark 5.3):
#   E[C] ~ n/tau,                      Var(C) ~ n/tau^3 Var(T)       (Proposition 5.2)
#   E[M] ~ n tau/2,                    Var(M) ~ n tau/4 Var(T)       (Proposition 5.4)
#   E[D] = tau - 1 + e_1/e_0 + o(1),   Var(D) = Var(T) + o(1)        (Proposition 5.6)
def asymptotic_moments(n, p):
    tau = tau_n(n, p)
    if p == 1/2:
        return {'C': (n/tau, n/tau**2), 'M': (n*tau/2, n*tau**2/4), 'D': (tau, tau)}
    e0, e1, e2 = (periodic_e(tau, p, l) for l in range(3))
    var_T = e2/e0 - (e1/e0)**2
    return {'C': (n/tau, n/tau**3*var_T), 'M': (n*tau/2, n*tau/4*var_T), 'D': (tau - 1 + e1/e0, var_T)}


ASYMPTOTE_NAMES = {
    'critical': {'C': ('n/W(n)', 'n/W(n)^2'), 'M': ('n W(n)/2', 'n W(n)^2/4'), 'D': ('W(n)', 'W(n)')},
    'subcritical': {'C': ('n/tau', 'n/tau^3 Var(T)'), 'M': ('n tau/2', 'n tau/4 Var(T)'),
                    'D': ('tau - 1 + e_1/e_0', 'Var(T)')},
}


# ----------------------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------------------

def _fmt_n(n):
    return str(n) if n < 1e7 else '1e{}'.format(round(math.log10(n)))


# Prints a table comparing values with a reference.
def report(title, ns, values, reference, names):
    print('\n' + title)
    print('    {:>6}{:>14}{:>14}{:>26}'.format('n', *names, '{}/{} - 1'.format(*names)))
    for n, v, r in zip(ns, values, reference):
        print('    {:>6}{:>14.6g}{:>14.6g}{:>+26.3e}'.format(_fmt_n(n), float(v), float(r), float(v/r - 1)))


def header(text):
    print('\n' + '=' * 100 + '\n' + text + '\n' + '=' * 100)


# ----------------------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------------------

MOMENTS = ('E[{}]', 'Var({})')


def check_regime(p, ns, large_ns):
    regime = 'critical' if p == 1/2 else 'subcritical'
    header('{} regime p = {}'.format(regime.capitalize(), p))

    print('\nExact moments vs saddle-point moments')
    exact = moments_mp(max(ns), p)
    saddle = [saddle_point_moments(n, p) for n in ns]
    for X in STATS:
        for k, moment in enumerate(MOMENTS):
            report(moment.format(X), ns, [exact[n][X][k] for n in ns], [s[X][k] for s in saddle],
                   ('exact', 'saddle'))

    print('\nSaddle-point moments vs asymptotes')
    saddle = [saddle_point_moments(n, p) for n in large_ns]
    asymptotes = [asymptotic_moments(n, p) for n in large_ns]
    for X in STATS:
        for k, moment in enumerate(MOMENTS):
            report('{} ~ {}'.format(moment.format(X), ASYMPTOTE_NAMES[regime][X][k]), large_ns,
                   [s[X][k] for s in saddle], [a[X][k] for a in asymptotes], ('saddle', 'asymptote'))


# Lemma B.3: tau_n = sqrt(2 log n/a) - 1/a + O(log log n/sqrt(log n)), with a = log((1-p)/p).
def check_tau(p, large_ns):
    a = math.log((1-p)/p)
    header('Asymptotics of tau_n for p = {} (Lemma B.3)'.format(p))
    print('\n    {:>6}{:>14}{:>24}{:>14}'.format('n', 'tau_n', 'sqrt(2 log n/a) - 1/a', 'difference'))
    for n in large_ns:
        tau = tau_n(n, p)
        approx = math.sqrt(2*math.log(n)/a) - 1/a
        print('    {:>6}{:>14.6g}{:>24.6g}{:>+14.3e}'.format(_fmt_n(n), tau, approx, tau - approx))


def main():
    parser = argparse.ArgumentParser(description='Numerical checks of the asymptotic means and variances.')
    parser.add_argument('--quick', action='store_true', help='only compute the exact moments up to n=250')
    args = parser.parse_args()
    ns = [50, 100, 250] + ([] if args.quick else [500, 1000])
    large_ns = [10**k for k in (3, 4, 8, 16, 32, 64, 128, 256)]
    for p in (1/2, 0.25):
        check_regime(p, ns, large_ns)
    check_tau(0.25, large_ns)


if __name__ == '__main__':
    main()
