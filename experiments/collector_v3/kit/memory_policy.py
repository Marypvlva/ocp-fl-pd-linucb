"""Timestamped weighted one-hot Joint. Resource ledgers/prices are never forgotten.

This is a research adaptation, not the linear-reward D-LinUCB regret theorem.
Pending observations do not reveal labels; late labels retain their generation age.
"""
from collections import deque
import math

import numpy as np
from model import Policy, SLA, CHANNELS, LEFT, RIGHT, EXTRA


class MemoryPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = self.s
        if s.mode not in ('joint', 'recent') or s.memory not in ('none', 'discount', 'window'):
            raise ValueError('Unsupported memory policy')
        if (type(s.discount) not in (int, float) or not math.isfinite(s.discount)
                or not 0 < s.discount <= 1 or type(s.window) is not int or s.window < 1
                or s.memory_bonus not in ('weighted', 'legacy')
                or s.feedback_rule not in ('legacy', 'eligible')):
            raise ValueError('Invalid memory settings')
        if type(s.delay) is not int or s.delay < 0 or not s.ridge > 0:
            raise ValueError('Invalid delayed-feedback geometry')
        self.design2 = np.zeros_like(self.design)
        self.done2 = np.zeros_like(self.done)
        self.age_mass = np.zeros_like(self.done)
        self.events = deque()
        self.clock = 0
        self.expired_feedback = 0
        self.live_event = None

    def _change(self, event, sign):
        age = self.t-event['time']
        w = self.s.discount**age if self.s.memory == 'discount' else 1.
        for ix, design, known, win in event['entries']:
            self.design[ix] += sign*w*design
            self.design2[ix] += sign*w*w*design
            self.done[ix] += sign*w*known
            self.done2[ix] += sign*w*w*known
            self.wins[ix] += sign*w*win
            self.age_mass[ix] += sign*w*age*known

    def _advance(self):
        dt = self.t-self.clock
        if dt:
            self.age_mass += dt*self.done
            if self.s.memory == 'discount':
                w = self.s.discount**dt
                for a in (self.design, self.done, self.wins, self.age_mass):
                    a *= w
                for a in (self.design2, self.done2):
                    a *= w*w
            if self.s.memory == 'window':
                # At decision t keep generation slots t-W,...,t-1.
                while self.events and self.events[0]['time'] < self.t-self.s.window:
                    event = self.events.popleft()
                    self._change(event, -1)
                    event['expired'] = True
                    event['entries'] = []
            self.clock = self.t
        for a in (self.design, self.done, self.wins, self.design2, self.done2, self.age_mass):
            if np.any(a < -1e-8):
                raise ArithmeticError('Negative weighted sufficient statistic')
            np.maximum(a, 0, out=a)

    def _mature(self, event):
        if event['expired']:
            self.expired_feedback += 1
            return
        self._change(event, -1)  # Remove still-unlabelled booked geometry, at its current age.
        c, roster, arm, _ = event['decision']
        readings, attempted, cutoff = event['feedback']
        entries = []
        for branch, g in enumerate((LEFT[arm], RIGHT[arm])):
            used = np.ones(self.r, bool) if branch == 0 else EXTRA[arm]
            obs = readings[:, :, branch]
            for i, d in enumerate(SLA):
                if self.s.feedback_rule == 'eligible':
                    # Eligibility is independent of the observed success value.
                    # Positive responses beyond an ineligible horizon are not pooled.
                    known = used[:, None] & attempted[:, :, branch] & (d <= cutoff[:, None])
                else:
                    known = used[:, None] & attempted[:, :, branch] & ((obs > 0) | (d <= SLA[c]))
                win = known & (obs > 0) & (obs <= d)
                ix = self._index(roster, g, i)
                entries.append((ix, known.astype(float), known.astype(float), win.astype(float)))
                self.recent[ix] = np.where(known, .65*win+.35*self.recent[ix], self.recent[ix])
        event['entries'] = entries
        self._change(event, 1)
        event['feedback'] = None

    def _score(self, c, roster):
        n, completed, wins = self.design[:, :, :, c], self.done[:, :, :, c], self.wins[:, :, :, c]
        mu = .5+(wins-.5*completed)/(self.s.ridge+n)
        if self.s.memory_bonus == 'weighted':
            bonus = self.s.alpha*np.sqrt(self.s.ridge+self.design2[:, :, :, c])/(self.s.ridge+n)
        else:
            bonus = self.s.alpha/np.sqrt(self.s.ridge+n)
        self.path_mean_prediction = np.clip(mu[self.b[:, None], roster, :], 0, 1)
        self.path_pooled_prediction = np.clip(mu.mean(axis=1), 0, 1)
        idx = (self.b[:, None], roster, slice(None))
        self.diagnostics = dict(
            label_mass=completed[idx].tolist(), design_mass=n[idx].tolist(),
            squared_label_mass=self.done2[:, :, :, c][idx].tolist(),
            effective_labels=np.divide(completed**2, self.done2[:, :, :, c],
                out=np.zeros_like(completed), where=self.done2[:, :, :, c]>0)[idx].tolist(),
            mean_label_age=np.divide(self.age_mass[:, :, :, c], completed,
                out=np.zeros_like(completed), where=completed>0)[idx].tolist(),
            path_bonus=bonus[idx].tolist(),
            pending_mass=np.maximum(0, n-completed)[idx].tolist())
        pred = self.recent[:, :, :, c] if self.s.mode == 'recent' else np.minimum(1, mu+bonus)
        if not self.s.use_current_roster:
            pred = pred.mean(axis=1)[:, None, :].repeat(self.k, axis=1)
        else:
            pred = pred[self.b[:, None], roster, :]
        union = 1-(1-pred[:, :, LEFT])*(1-pred[:, :, RIGHT])
        union[:, :, :CHANNELS] = pred
        return np.prod(union, axis=1)

    def decide(self, c, roster, forced_arm=None):
        if self.live_event is not None:
            raise ValueError('Observe the preceding decision exactly once')
        self._advance()
        arm = super().decide(c, roster, forced_arm)
        entries = []
        for branch, g in enumerate((LEFT[arm], RIGHT[arm])):
            used = np.ones(self.r, bool) if branch == 0 else EXTRA[arm]
            ix = self._index(roster, g, c)
            mask = np.broadcast_to(used[:, None], (self.r, self.k)).astype(float).copy()
            # Parent already added immediate geometry, but not the squared moment.
            self.design2[ix] += mask
            entries.append((ix, mask, np.zeros_like(mask), np.zeros_like(mask)))
        event = dict(time=self.t, decision=self.last, entries=entries, expired=False)
        if self.s.memory == 'window':
            self.events.append(event)
        self.live_event = event
        self.last_decision['memory'] = dict(mode=self.s.memory, discount=self.s.discount,
            window=self.s.window, bonus=self.s.memory_bonus, feedback_rule=self.s.feedback_rule,
            expired_feedback=self.expired_feedback, **self.diagnostics)
        return arm

    def observe(self, readings, reward, attempted=None, *, observation_horizon_ms=None):
        event = self.live_event
        if event is None:
            raise ValueError('Feedback needs an unmatched decision')
        readings = np.asarray(readings)
        attempted = np.ones_like(readings, bool) if attempted is None else np.asarray(attempted, bool)
        if readings.shape != (self.r, self.k, 2) or attempted.shape != readings.shape:
            raise ValueError('Invalid path feedback shape')
        if not np.isfinite(readings).all():
            raise ValueError('Nonfinite path feedback')
        cutoff = np.full(self.r, SLA[event['decision'][0]]) if observation_horizon_ms is None else np.broadcast_to(observation_horizon_ms, (self.r,)).copy()
        if not np.isfinite(cutoff).all() or np.any(cutoff < 0):
            raise ValueError('Invalid observation horizon')
        cutoff = np.minimum(cutoff, SLA[event['decision'][0]])
        event['feedback'] = readings.copy(), attempted.copy(), cutoff
        self.pending.append((self.t+self.s.delay, event))
        self.live_event = None
        self.t += 1
