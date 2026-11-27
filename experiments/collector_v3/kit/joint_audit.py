"""Read-only, action-certified audit of Joint histories; no counterfactual rewards."""
import argparse
import json
from pathlib import Path

import numpy as np
from model import Policy, Settings, SLA, ARMS, USE, EXTRA
from gateway_study import caps
from program import read_plan
from analysis_program import table, write_json


class HistoryAudit(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_times = np.zeros_like(self.done)
        self.pre_shift_mass = np.zeros_like(self.done)

    def _mature(self, payload):
        before = self.done.copy()
        super()._mature(payload)
        delta = self.done - before
        self.source_times += delta * payload[7]
        if payload[7] < 40:
            self.pre_shift_mass += delta

    def observe(self, *args):
        super().observe(*args)
        if self.pending and self.pending[-1][0] == self.t - 1 + self.s.delay:
            due, payload = self.pending[-1]
            self.pending[-1] = due, payload + (self.t - 1,)


def analyze(directory, output):
    from deploy.timeweb.summarize import audit
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or directory in output.parents:
        raise ValueError('Never write into original data')
    output.mkdir(parents=True, exist_ok=False)
    roots = sorted(p.parent for p in directory.rglob('plan.json'))
    if (directory / 'plan.json').exists():
        roots = [directory]
    if not roots:
        raise ValueError('No experiment plans')
    rounds, episodes, integrity = [], [], []
    for root in roots:
        plan, digest = read_plan(root / 'plan.json')
        result = audit(root)
        if not result['complete']:
            raise ValueError('Incomplete data: ' + str(root))
        integrity.append(dict(series=str(root.relative_to(directory)), plan_sha256=digest, **result))
        SLA[:] = plan['windows_ms']
        for index, seed in enumerate(plan['seeds']):
            for mode in plan['modes']:
                settings = Settings(**plan['policy_specs'][mode])
                if settings.memory != 'none' or settings.feedback_rule != 'legacy':
                    raise ValueError('This audit reconstructs v0.7.0 only; use replay_program for new memory policies')
                if settings.mode == 'article_pd':
                    from article_policy import ArticlePolicy
                    cls = ArticlePolicy
                else:
                    cls = HistoryAudit
                policy = cls(settings, 1, plan['population'], plan['k'], plan['horizon'],
                    plan.get('mirror_limit', plan['horizon']//4), plan.get('policy_tie_seed', 53),
                    caps(index, plan['horizon'], plan.get('site_fraction', .2))[None, :],
                    np.array([plan['calibration']['prices'][str(index%4)]]))
                files = sorted((root/f'{seed}_{mode}').glob('round-*.json'))
                episode = []
                for file in files:
                    r = json.loads(file.read_text())
                    t = r['round']-1
                    c = plan['windows_ms'].index(r['window_ms'])
                    forced = plan.get('forced_arms', {}).get(str(seed))
                    arm = int(policy.decide(c, np.array([r['selected']]),
                        forced_arm=forced[t] if forced else None)[0])
                    if list(ARMS[arm]) != r['gateways']:
                        raise ValueError(f'Action mismatch {file}; cannot certify replay')
                    d = policy.last_decision
                    for key in ('predictions', 'penalized_scores', 'price_before',
                                'gateway_prices_before', 'mirror_spent', 'gateway_spent'):
                        if key in d and key in r['policy_decision']:
                            a, b = d[key], r['policy_decision'][key]
                            if a != b:
                                # Numerical tolerances do not excuse action mismatches.
                                if not np.allclose(np.array(a, float), np.array(b, float),
                                                   atol=1e-9, rtol=1e-9, equal_nan=True):
                                    raise ValueError(f'Diagnostic mismatch {key}: {file}')
                    base = np.array(d['predictions'][0])
                    feasible = np.array(d['feasible_arms'][0], bool)
                    gm = np.array(d['effective_gateway_prices'][0]) @ USE.T
                    mm = d['effective_mirror_price'][0] * EXTRA
                    selected = np.array(r['selected'])
                    age = old = pending = None
                    if isinstance(policy, HistoryAudit):
                        mass = policy.done[0, selected, :, c]
                        den = mass.sum()
                        age = float((t*mass-policy.source_times[0, selected, :, c]).sum()/den) if den else None
                        old = float(policy.pre_shift_mass[0, selected, :, c].sum()/den) if den else None
                        # Current action has just booked k designs; exclude that new design.
                        pending = float((policy.design-policy.done).sum()-plan['k']*len(ARMS[arm]))
                    samples = np.array(r['path_readings'])
                    mask = np.array(r['path_observed'], bool)
                    receipts = r['metrics'].get('collector_ms', {})
                    coverage = all(any(str(w) in receipts.get(str(g), {}) for g in r['gateways'])
                                   for w in r['selected'])
                    mean_coverage = None
                    if d.get('path_mean_prediction') is not None:
                        path = np.array(d['path_mean_prediction'][0])
                        if settings.mode == 'recent':
                            path = policy.recent[0, selected, :, c]
                        per_worker = 1-np.prod(1-path[:, list(ARMS[arm])], axis=1)
                        mean_coverage = float(np.prod(per_worker))
                    phase = t//40
                    row = dict(series=str(root.relative_to(directory)), scenario=plan.get('program',{}).get('scenario',{}).get('id'),
                        seed=seed, policy=mode, round=t+1, phase=phase, accepted=int(r['accepted']),
                        arm=arm, gateways=r['gateways'], mirror_spent=int(policy.spent[0]),
                        mirror_remaining=int(policy.budget-policy.spent[0]),
                        site_remaining=(policy.caps-policy.gateway_spent)[0].tolist(),
                        feasible_actions=int(feasible.sum()), best_prediction_infeasible=bool(not feasible[base.argmax()]),
                        price_changes_rank=bool(base[arm] < base[feasible].max()-1e-7),
                        mirror_changes_rank=bool(base[np.argmax(np.where(feasible, base-mm, -np.inf))] < base[feasible].max()-1e-7),
                        site_changes_rank=bool(base[np.argmax(np.where(feasible, base-gm, -np.inf))] < base[feasible].max()-1e-7),
                        mirror_price=d['effective_mirror_price'][0], site_prices=d['effective_gateway_prices'][0],
                        mean_label_age=age, pre40_label_fraction=old, pending_designs=pending,
                        age_scope='accumulated_labels_not_effective_EWMA_weights',
                        round_bytes=r['telemetry']['app_tx_bytes'], spent_bytes=r['byte_ledger']['spent'],
                        denied_frames=r['byte_ledger']['denied_frames'], projection_started=r['metrics']['projection_started'],
                        coverage=int(coverage), mean_coverage_prediction=mean_coverage,
                        optimistic_prediction=float(base[arm]),
                        mean_coverage_brier=(mean_coverage-coverage)**2 if mean_coverage is not None else None,
                        optimistic_coverage_brier=float((base[arm]-coverage)**2),
                        coverage_without_accept=int(coverage and not r['accepted']),
                        full_round_s=r['total_round_s'], observation_horizon_ms=r['metrics'].get('collector_phase_budget_ms'),
                        nominal_window_ms=r['window_ms'])
                    policy.observe(samples, np.array([r['accepted']]), mask)
                    rounds.append(row)
                    episode.append(row)
                for phase in sorted({x['phase'] for x in episode}):
                    seq = [x for x in episode if x['phase']==phase]
                    eligible = [x['round']-(40*phase) for x in seq if x['accepted']]
                    # Missing first success is right censored, never silently dropped.
                    out = dict(series=seq[0]['series'], scenario=seq[0]['scenario'], seed=seed, policy=mode,
                        phase=phase, rounds=len(seq), accepted=sum(x['accepted'] for x in seq),
                        first_success_offset=min(eligible) if eligible else None,
                        recovery_censored=not bool(eligible), recovery_observation_rounds=len(seq),
                        mirror_start=seq[0]['mirror_remaining']+int(EXTRA[seq[0]['arm']]),
                        site_end=seq[-1]['site_remaining'], mirror_end=seq[-1]['mirror_remaining'],
                        denied_end=seq[-1]['denied_frames'])
                    for key in ('price_changes_rank','best_prediction_infeasible','mirror_changes_rank',
                                'site_changes_rank','projection_started','coverage_without_accept'):
                        out[key] = sum(x[key] for x in seq)
                    for key in ('mean_label_age','pre40_label_fraction','mean_coverage_brier','optimistic_coverage_brier'):
                        vals = [x[key] for x in seq if x[key] is not None]
                        out[key] = float(np.mean(vals)) if vals else None
                    episodes.append(out)
    table(output/'rounds.csv', rounds, latex=False)
    table(output/'phases.csv', episodes)
    write_json(output/'integrity.json', integrity)
    write_json(output/'scope.json', dict(rounds=len(rounds), plans=len(roots),
        status='exact_action_and_stored_state_reconstruction',
        caution='Observed policy histories only; no counterfactual reward estimates or iid-round inference. Pilot seeds, host/time confounding and outcome-dependent censoring remain.'))
    return episodes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    analyze(a.directory,a.out)

if __name__=='__main__':
    main()
