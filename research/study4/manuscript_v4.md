# Resource-Constrained Scheduling of Verifiable Computations on Edge Nodes

Research manuscript revision of 21 November 2026. The author list and complete bibliography require editorial completion. Cloud measurements are pending.

## Abstract

Auditable client selection fixes the participants of a federated computation before execution. A scheduler subsequently chooses replication and verification mechanisms under limited resources. We combine OCP-FL cohort certificates with a PD-LinUCB-based scheduler and a latent fault filter. We evaluate the contribution of residual probability calibration, exploration and resource prices using common information, feedback timing and action costs. A locally frozen study comprises 7,168 episodes across eight conditions and 128 new seeds per condition. On three prespecified reference conditions, the full scheduler improves on the same-filter, same-price control by 0.70 accepted updates per 480 rounds, with a 95% paired interval of [0.47, 0.92]. The prespecified practical superiority threshold of one additional update is not met. Under moderate resource scarcity, adaptive pricing adds 40.15 updates [38.27, 42.02] relative to disabling prices while retaining hard feasibility guards. Per-condition comparisons identify both gains and a loss when resources are ample. A hybrid evaluation driven by historical reachability traces provides no advantage over a same-filter control. These findings establish the role of budget allocation and delimit the additional benefit of learned calibration. Local TCP/EVM acceptance checks cover reproducible 16-client cohorts and all subsets of four failed aggregation replicas. Multi-host performance remains to be evaluated.

## 1. Problem and contribution

The two foundations have complementary roles. OCP-FL supplies a verifiable cohort certificate before the execution decision. PD-LinUCB supplies a sequential decision rule with delayed learning and resource prices. The scheduler must respect the certified cohort while choosing how to execute and verify its computation.

Previous experiments in this project showed an advantage over an earlier greedy control. That comparison changed the representation of fault history and the price update rule alongside the learning algorithm. We therefore evaluate an explicit decomposition in which the compared policies share the same latent filter and feedback interface. We also construct resource profiles where a clairvoyant planner cannot recover every recoverable round within the quotas.

The contribution is an empirical characterization of when calibration and pricing are useful in this verified execution pipeline. The study does not establish universal superiority or a regret theorem for the composed filtered model.

## 2. Execution, reward and information

Each round has one certified cohort. The available actions are F+P, F+B, R+P and R+B. F uses primary aggregation replicas, R also uses reserve replicas, P verifies random projections, and B uses signed full updates and recomputation. A round earns reward one if its verified aggregate is accepted. The reward counts an aggregate model update, regardless of cohort size.

The study charges known action credits (1, 2, 2, 3) and normalized traffic costs (1.00, 1.40, 1.55, 1.95). These traffic values are protocol cost proxies. Actual application bytes are separately measured on the TCP implementation. The scheduler reserves enough credits and normalized traffic to make the cheapest action feasible in every remaining round.

Feedback records whether computation, aggregation and verification actually executed. An unexecuted stage provides an unknown label. Primary-lane failures E and missing projection responses M update the filter at round completion. Reserve failures S and missing full updates U update their reliability counts with a two-round delay. Residual ridge regressions receive labels at the same delay. This two-channel information schedule is an explicit assumption of the experiments.

The implementation exposes additional information beyond the pooled indicators. A coordinator can identify the address of an unanswered worker RPC. We therefore regard restricted pooled feedback and named response feedback as distinct observation regimes. Uniform cohort selection alone does not ensure diverse observations for every action after policy selection and stage censoring.

## 3. Scheduler and matched controls

AF-L maintains approximate per-client fault beliefs with a bank of Markov persistence parameters and a noisy-OR pooled likelihood. It uses the filter probabilities as offsets for residual ridge regressions. Action values compose the estimated stage probabilities according to the verification protocol. Selection adds a delta-method exploration bonus and subtracts one price per known-cost resource. Prices are re-solved against the empirical history and remaining target rates.

The matched filter-only control uses the same beliefs, delayed S/U counts, action composition, quotas, price solver and information masks. It fixes residual coefficients to zero and sets alpha to zero. A residual-greedy control enables the residual regressions with alpha zero. The full AF-L uses alpha=0.05 and ridge=10. Further controls replace re-solved prices with gradient updates at eta=0.05, disable prices while retaining hard guards, or disable the second price while retaining both guards. The price-free matched contrast uses the residual-greedy policy, so that exploration is identical.

The noisy-OR posterior equations are exact for one update under an independent Bernoulli prior with independent leak. Shared generator states, estimated base rates, persistence selection, posterior factorization and censoring introduce additional approximation. The exploration bonus omits uncertainty in the learned filter. Confidence or regret guarantees for the original PD-LinUCB setting do not automatically transfer to this model.

## 4. Design and reproducibility

The Study 4 plan and source snapshot were stored locally before execution on 20 November 2026. There is no independent external registration timestamp. Seeds 610000–610127 are new relative to development and Study 3. Each episode has 480 rounds. Eight conditions and seven policies produce 7,168 episodes, totaling 3,440,640 simulated rounds. The statistical unit is a paired seed, not an individual round.

The primary comparison is AF-L minus filter-only averaged across S1, D2 and S10 within each seed. We report a two-sided Wilcoxon test and a mean interval from 10,000 seed-block bootstrap samples. Statistical support requires a positive mean and p<0.05. Practical superiority additionally requires the lower 95% bound to exceed one update. The secondary family comprises eight per-condition AF-L versus filter-only tests with Holm correction. Other component contrasts are exploratory. Sample size was fixed using development standard deviations and a normal-approximation calculation for detecting a one-update difference. This calculation does not guarantee power for the separate practical-margin criterion.

Three new resource profiles share identical physical worlds. Their credit ratios are 1.10, 1.25 and 3.00, and traffic ratios are 1.0475, 1.11875 and 1.95. We compute clairvoyant optima under both quotas, under each quota separately and without quotas. A shifted-sequence condition circularly shifts the complete fault indicators by half the horizon relative to the current rosters. This preserves the realized fault sequence and breaks its original roster association. A final condition adds unreliable reserve and full-upload remedies.

The earlier known-model reference has been repaired for all 864 original episodes where it was defined. It now masks unexecuted stages and incorporates delayed S/U observations through retrospective filtering. Its largest change in a condition mean is −0.66 updates. The reference remains privileged through known group schedules and transition laws, and its allocation policy remains a price heuristic. It is distinct from the exact clairvoyant optimum.

## 5. Results

The primary difference is +0.695 updates per 480 rounds [0.469, 0.924], with Wilcoxon p=5.39×10⁻⁸. This is approximately 0.145 percentage points of all attempts. The practical-margin criterion is not satisfied. Of 128 seed-level average differences, 86 are positive, ten zero and 32 negative.

| Condition | AF-L | Matched filter | Difference [pointwise 95% interval] |
|---|---:|---:|---|
| S1 | 454.96 | 454.78 | +0.18 [−0.12, +0.48] |
| D2 | 456.13 | 455.73 | +0.41 [+0.13, +0.67] |
| S10 | 409.34 | 407.84 | +1.50 [+0.96, +2.05] |
| Tight | 349.20 | 348.46 | +0.73 [+0.25, +1.23] |
| Moderate | 378.53 | 377.41 | +1.13 [+0.29, +1.97] |
| Ample | 449.91 | 450.91 | −1.00 [−1.16, −0.84] |
| Shifted sequence | 361.44 | 358.52 | +2.92 [+2.15, +3.70] |
| Noisy upload | 359.84 | 359.55 | +0.29 [−0.48, +1.05] |

Five positive per-condition differences survive Holm correction. The ample-budget condition has a significant negative difference. S1 and noisy-upload differences are inconclusive. In the ample condition, the filter-only control reaches the clairvoyant optimum. Added calibration and exploration can therefore impose a measurable cost in a setting where the simple remedy choice already suffices.

In the tight condition, the jointly imposed quotas reduce the clairvoyant optimum relative to unlimited resources in all 128 worlds. Removing only the credit quota improves the optimum in 127 worlds, while removing only traffic never changes it. In the moderate condition, the joint quotas reduce the optimum in 98 worlds. Removing credit improves it in 48 worlds and removing traffic improves it in 41. Geometric independence of quota bounds is therefore insufficient to establish empirical influence of both resources.

With the same residual learning and alpha=0, adaptive prices add 21.70 updates [20.64, 22.77] in the tight condition and 40.15 [38.27, 42.02] in the moderate condition relative to zero prices. Both policies retain the same hard quotas. With the same residual learning and alpha=0.05, re-solving adds 3.92 updates [2.98, 4.91] over gradient price updates in the moderate condition. This comparison concerns eta=0.05. A tuned-gradient claim would require separate development data. Adding the second price while retaining both feasibility guards contributes +0.66 [0.02, 1.30] in that condition.

On the three reference conditions, the UCB increment over residual-greedy is +0.21, +0.04 and +0.03. Its exploratory increment in moderate scarcity is +0.48 [0.04, 0.93]. These data provide limited support for making exploration the central contribution.

## 6. External trace and implementation checks

A hybrid exploratory evaluation uses the PlanetLab All Pairs Ping reachability trace distributed in the Repository of Availability Traces. We select seven disjoint groups of 18 nodes with sufficient boundary coverage and four five-day windows per group at 15-minute sampling intervals. Sixteen records provide client-route or projection-response risk. Two records provide alternative-path and full-upload failure conditions. The mapping from recorded availability to the computation remedies is a modeling assumption. The measurements do not directly record our verification protocol.

Across 84 episodes, AF-L accepts 394.71 updates, filter-only 394.93 and a control with named projection responses 396.29. Node groups are disjoint but may share correlated infrastructure and measurement times. We make no broad statistical generalization from this small hybrid study. It demonstrates that the synthetic-generator advantage does not transfer automatically. [Trace repository and dataset documentation](https://pbg.web.engr.illinois.edu/availability/)

Local implementation checks use real TCP roles and the test EVM. Reproducible additional test accounts permit paired 16-client cohorts. Both policies receive identical certificate sequences. Four short 16-client episodes, 36 rounds in total, pass individual audits across the initial and strengthened orchestration versions. All 16 subsets of failed aggregation replicas produce the expected fragment availability. Feedback schema 3 recognizes successful recovery from mixed primary and reserve replicas. Historical schema 2 journals preserve their recorded interpretation. The full 187-test suite passes, alongside checks rejecting missing episodes and altered configurations.

## 7. Scope and remaining evaluation

The evidence supports budget allocation as an important mechanism in this pipeline. The additional value of residual calibration is conditional on resource pressure and model error. The shifted-sequence gain may reflect correction of a misspecified attribution model, so it cannot establish client fault localization by itself. Gains expressed as regret reduction must be accompanied by their absolute accepted-update differences.

The next physical experiment is a multi-host acceptance and variance pilot with a matched filter control, new seeds and sequential execution. The pilot will measure actual application bytes, complete-cycle latency including OCP-FL, deadline misses, verified updates per minute and learning quality. The available deployment controls normalized quotas across hosts and measures bytes per role. A shared hard byte quota across hosts requires additional reservation logic. The present work contains no completed cloud performance study.

The synthetic generator also enforces U implies M and makes reserve execution harmless to a healthy primary path. Consequently, R+B weakly dominates every other action in realized success and attains the unrestricted optimum in all 1,024 generated worlds. Independent full-upload and projection failures, together with reserve waiting-time costs, are a necessary next model extension. The TCP protocol permits a full-upload failure without a projection failure.

A future algorithmic study could compare the base filter and residual calibration using only prediction errors available from past executed stages. Such a selection mechanism would need its own development split and frozen evaluation. Natural client and shared-route failures, useful prediction beyond recent route telemetry, and the effect on final model quality remain necessary targets for a deployment claim.

## Foundation references

The supplied OCP-FL manuscript, *On-Chain Provenance for Federated Learning — Verifiable Random Client Scheduling against Bias and Poisoning*, defines the auditable selection foundation. The supplied PD-LinUCB manuscript defines the primal–dual contextual-bandit foundation with budget constraints and delayed rewards. Their bibliography and publication details should be verified in the final submission. Earlier comparative results are retained as historical Study 3 data, with the repaired reference and changed interpretation documented separately.
