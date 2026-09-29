# Mathematical specification and limits

This appendix belongs to `CODEX_FINAL_EXECUTION_EN.md`. It specifies proposed operators; none has been measured on the user's scenes in this instruction-generation turn. `math_reference.py` is a small synthetic reference, not production data plumbing.

## 1. Why the experiment is justified, without presupposing the answer

For a three-source arithmetic pool and two classes g,w, the region expert can overcome the N/Q preference for w only if

`p_R(g)-p_R(w) > [p_N(w)-p_N(g)] + [p_Q(w)-p_Q(g)]`.

The left side is at most 1. Therefore a right side >1 prevents even a maximally confident region distribution from beating w. This is an exact bound on a particular pooling operator, NOT a proof that this explains the empirical OVR deficit. The GT-dependent count is an evaluation-only diagnostic after labels are locked. A right side <=1 does not establish a successful correction.

## 2. Common-temperature paired residual

For common paired views, define `r=(s_O-s_F)/T_F`. Because the log-softmax normalizers are class-independent,

`r(a)-r(b) = log[p_O^common(a)/p_F^common(a)] - log[p_O^common(b)/p_F^common(b)]`.

Thus using raw score differences at a common T gives the same relative residual as the ratio of common-temperature probabilities. Equal score vectors imply an identically zero update. If separately calibrated temperatures are used instead, equal score vectors need not imply zero residual. That is why R3_CALDELTA is a diagnostic, not the cleanest attribution to paired representation change.

Pairing controls source image, mask, target, view weights, prototype policy and successful request set. The model text towers may also change; compare their saved prototype arrays before naming the difference a visual-only increment. Do not subtract FC and OVR latent feature vectors: class-aligned cosine scores are the common comparison space.

## 3. Local KL update: derivation and safe claims

Let `p0` be B's positive probability vector and C its fixed candidate set. Let `q=sum_C p0`, `pi0=p0[C]/q`. Solve over the simplex of C:

`min_pi KL(pi || pi0) - eta * <pi,r>`.

The Lagrangian first-order condition is

`log(pi_c/pi0_c) + 1 - eta*r_c + nu = 0`,

so `pi*=softmax(log(pi0)+eta*r)`. Set `pnew[C]=q*pi*` and keep outside values equal to p0. This is the unique minimizer when pi0 is strictly positive. Its properties are:

- eta=0 or residual constant within C returns p0;
- outside probabilities remain exactly unchanged by design;
- total candidate mass remains q;
- local odds are multiplied by `exp(eta*(r_a-r_b))`;
- no guarantee of preserving a correct top-1, improving calibration, or improving AP follows.

Preserving probability mass does not preserve semantic point mass, class frequency, instance count per class or ranking after labels change. Do not conflate these meanings of mass.

Global ratio update is the same variational operation on the full class set. R0 blend and R1 fourth-source pooling are essential simple comparisons; the KL expression itself is not a novelty claim.

## 4. Derivative of area-normalized directional evidence

For fixed observations and positive normalized area weights,

`a=sum_e alpha_e f_e`, `v=a/||a||`, `d_ab=(t_a-t_b)^T v / T`.

Differentiating the normalization map yields

`grad_(f_e) d_ab = alpha_e/(T||a||) * (I-vv^T)*(t_a-t_b)`.

The numerical reference checks this derivative against central finite differences. Original saved view norms are retained. The derivative is evaluated on a float64 surrogate; N0's original FP32 readout and tie behavior are still reproduced with its own original functions. Derivative and frozen-score numerical paths must not be confused.

Relative isotropic noise with expected squared norm proportional to `||f_e||²` produces loading `L_e=||f_e||*J_e/sqrt(d)`. This is a declared perturbation model, not an estimate from true semantic errors. A visually stable but systematically wrong expert can still be assigned low proxy variance. This is a reason to test the proxy, not a guarantee it will work.

## 5. Kernel and covariance positive semidefiniteness

For each known support mask, define `b_e=vec(M_e)/sqrt(|M_e|)`. Embed b into the block specified by the same original image identity and same vision-checkpoint coordinate system. Missing-mask atoms have a private orthogonal unit basis instead. Then

`Kobs[e,f]=<b_e,b_f>`

is a Gram matrix and hence PSD. Known different frame/checkpoint blocks have zero modeled overlap; this does not establish true independence. The normalized mask overlap is an inner-product kernel, not intersection-over-union.

For compatible feature-coordinate blocks,

`Sigma0[m,n]=sum_(e,f) Kobs[e,f] * <L_me,L_nf>`.

This is also a Gram matrix (equivalently use Kronecker features), hence PSD. Adding `0.1*vbar*I` makes it SPD for positive vbar. It is unsafe to manually insert arbitrary pair correlations, since that would not preserve PSD.

With N/Q sharing a visual space and F in a separate block, the only possible inter-source covariance in this first implementation is N/Q. This is intentionally narrower than a complete learned model-error covariance. Do not claim to remove FC/native shared pretraining bias, long-range temporal correlation or all repeated information.

Exact duplicate-source invariance is implemented by collapsing verified equivalent source records **before** this operator and the diagonal floor. Duplicating an alias of the same frozen evidence must not increase source weight. Do not collapse merely equal top-1 labels or close scores. The baseline p0 remains fixed in this diagnostic.

## 6. Simplex minimum variance and graph recovery

For SPD S, the source-weight problem is convex:

`min_{w>=0,1^T w=1} w^T S w`.

On an active subset A, the equality-constrained solution is

`w_A = solve(S_A,1) / (1^T solve(S_A,1))`.

Enumerating every nonempty A for at most three sources and keeping feasible solutions includes the constrained optimum, including vertex solutions. No learned covariance and no source-unbiasedness claim is implied. Minimum proxy variance is not minimum semantic risk.

Let H be the oriented complete class-incidence matrix and W positive edge weights. The graph problem is

`||W^(1/2)(Hu-dhat)||² + K||u-u0||²`.

Its unique solution satisfies `(H^T W H+K I)u=H^T W dhat+K u0`. If u0 is centered and every edge is a difference, the solution is centered up to roundoff. For uniform W and consistent equal-source differences generated by centered zbar, `H^T H=K I-11^T`; therefore

`u=(zbar+u0)/2` on the zero-sum subspace.

This identity is tested and motivates D1_ANCHORED. Without that control, an apparent D benefit could be entirely due to the reference anchor.

Pairwise log odds cancel vocabulary normalizers for an existing pair. Adding classes changes the whole graph, its ridge K and the final argmax. **There is no claim of full vocabulary-expansion invariance.**

## 7. Shuffled-dependence control

The SPD covariance has only one off-diagonal source pair N/Q. Let its correlation be rho with |rho|<1. Permuting rho across class pairs while keeping each pair's diagonal variances produces another SPD 2x2 block, independent of F. It preserves marginal proxy variances but breaks the alignment of class-pair correlation.

D4_SHUFFLED is not an image-shuffle control and does not shuffle a true measured error covariance. It tests whether the estimated class-pair dependence alignment helps beyond generic correlated weighting. Constant rho or no N/Q evidence gives an identity control; report this degeneration.

## 8. Important non-results and failure modes

| Observed situation | Correct interpretation |
|---|---|
| Eta=0 wins | No selected effective residual; still publish active-grid outcomes. |
| D4=D3, no modeled cross support | Off-diagonal mechanism not exercised; do not claim a successful dependency correction. |
| D4 beats logpool but not D1_ANCHORED | Anchor or ordinary blending may explain the benefit. |
| D4_SHUFFLED is as good as D4 | Specific dependence alignment not supported. |
| R3CAL wins but common-T R3 does not | Calibration changes may be material; not a clean visual-delta result. |
| R3 helps only under missing-pair fallback | Separate fallback preservation from actual intervention. |
| Candidate contains wrong classes | Local mass conservation does not prevent wrong labels. |
| Few views / one view | Derivative proxy exists, but does not estimate empirical variability; record support. |
| Shared N/Q images rare | The rationale for this covariance kernel may be weak; do not broaden it after seeing test labels. |
| New official rankings alter AP | Report frozen-rank control; never use a new ranking rule just to improve AP. |
| A simple two-stage blend wins | Prefer the simpler measured method; no forced novelty claim. |

## 9. Deferred acquisition equation

The previously discussed conditional precision increment

`Delta I = (1-c^T Sigma^-1 1)^2 / (v-c^T Sigma^-1 c)`

holds only for a particular common-latent, unit-loading linear Gaussian observation model with SPD joint covariance. It is not an AP gain and does not justify using unencoded visual features in a query decision. This study neither trains its predictor nor changes acquisition. Existing information-gain/sensor-placement theorems do not transfer automatically to the proposed proxy.

## 10. Verified scope of the bundled check

`MATH_CHECK_RESULTS.json` reports synthetic checks of the reference operators: residual mass, null cases, Jacobian, PSD construction, small QP, graph identity and shuffled correlation. It does **not** validate repository integration, actual source pairing, ground-truth separation, calibration, benchmark metrics, or publication. Codex must finish the task-specific real-data checks and evaluation separately.
