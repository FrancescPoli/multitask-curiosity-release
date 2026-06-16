
import numpy as np
import gym
from neurogym import spaces
import neurogym as ngym
from neurogym.core import TrialEnv

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_dist(original_dist):
    """Get the distance in periodic boundary conditions"""
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))

def _gaussianbump(loc, theta, strength):
    dist = _get_dist(loc - theta)  # periodic boundary
    dist /= np.pi / 8
    return 0.8 * np.exp(-dist ** 2 / 2) * strength

def _get_timing(default, override):
    t = default.copy()
    if override:
        t.update(override)
    return t

# ---------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------

class PoliEnv(TrialEnv):
    def __init__(self, dt=100, dim_ring=16):
        super().__init__(dt=dt)
        self.dim_ring = dim_ring
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.choices = np.arange(dim_ring)
        self.rng = np.random.RandomState()
        self.rng = np.random.RandomState()
        # self.baseline = 0.2 # Removed: Granular tuning per task

    def _init_gt(self):
        tmax_ind = int(self._tmax / self.dt)
        if tmax_ind > 0:
            self.gt = np.zeros([tmax_ind] + list(self.action_space.shape),
                           dtype=self.action_space.dtype)
            self._gt_built = True

    def set_gt(self, value, period=None):
        if not hasattr(self, '_gt_built') or not self._gt_built:
            self._init_gt()
            
        if period is None:
            self.gt[:] = value
        elif isinstance(period, str):
            if period in self.start_ind:
                idx_start = self.start_ind[period]
                idx_end = self.end_ind[period]
                self.gt[idx_start:idx_end] = value
                
    def add_baseline(self, period, where):
         """Add baseline to observation."""
         if hasattr(self, 'baseline'):
              self.add_ob(self.baseline, period=period, where=where)

# ---------------------------------------------------------------------
# 1. PoliReach
# Matches yang19.go, anti, etc.
# Timings verified: 
# Standard: fix 500, stim 500, delay 0, dec 500
# Delayed:  fix 500, stim 500, delay 500, dec 500
# RT:       fix 500, stim 500, delay 0, dec 500 (Stim+Dec are response periods)
# ---------------------------------------------------------------------
class PoliReach(PoliEnv):
    def __init__(self, dt=100, anti=False, category=False, context=False, 
                 reaction=False, delay=False, dim_ring=16, timing=None):
        super().__init__(dt=dt, dim_ring=dim_ring)
        self.anti = anti
        self.category = category
        self.context = context
        self.reaction = reaction
        self.delay = delay
        
        # Exact Yang19 Timing (Variable)
        self.timing = {
            'fixation': lambda: self.rng.uniform(300, 700),
            'stimulus': lambda: self.rng.uniform(500, 1500),
            'delay': 0,
            'decision': 500
        }
        if self.delay:
             self.timing['delay'] = 500
             
        if timing:
            self.timing.update(timing)

        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        self.abort = False

        self.n_rings = 2 # Always 2 rings (32 dims) to match Unified Space
        name = {'fixation': 0}
        for r in range(self.n_rings):
            name[f'stimulus_mod{r+1}'] = range(1 + r*dim_ring, 1 + (r+1)*dim_ring)
            
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + self.n_rings * dim_ring,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1 + dim_ring, name=name)

    def _new_trial(self, **kwargs):
        trial = {
            'anti': self.anti,
            'category': self.category,
            'context': self.context,
        }
        trial.update(kwargs)

        i_theta = self.rng.choice(self.choices)
        theta = self.theta[i_theta]
        trial['theta'] = theta
        
        # Standard Periods for all (RT just changes behavior/GT)
        if self.reaction:
             # RT: Fixation -> Stimulus. Response allowed during Stimulus.
             # Trial ends immediately after Stimulus.
             periods = ['fixation', 'stimulus']
        else:
             periods = ['fixation', 'stimulus', 'delay', 'decision']
             
        self.add_period(periods)
        
        fix_periods = ['fixation', 'stimulus', 'delay']
        if self.reaction:
             fix_periods = ['fixation'] # End fix requirement at Stimulus onset
        if self.reaction:
             fix_periods = ['fixation'] # Only enforce/show fix during fixation period
             
        self.add_ob(1, period=fix_periods, where='fixation')
        
        # No Baseline for Go tasks
        # self.add_baseline(period=periods, where='stimulus_mod1')
        # self.add_baseline(period=periods, where='stimulus_mod2')

        stim_bump = _gaussianbump(theta, self.theta, 1.0)
        stim_period = 'decision'
        if self.reaction:
             # Stimulus is presented during 'stimulus' period (500ms).
             # Start of stimulus is the Go cue.
             stim_period = 'stimulus'
        else:
             # Standard/Delay: Stimulus presented during 'stimulus' period.
             stim_period = 'stimulus'
        
        if self.context:
            i_dist = self.rng.choice(self.choices)
            while i_dist == i_theta:
                 i_dist = self.rng.choice(self.choices)
            dist_theta = self.theta[i_dist]
            dist_bump = _gaussianbump(dist_theta, self.theta, 1.0)
            
            self.add_ob(stim_bump, period=stim_period, where='stimulus_mod1')
            self.add_ob(dist_bump, period=stim_period, where='stimulus_mod2')
        else:
            # Randomly choose Ring 1 or Ring 2
            mod = self.rng.choice([1, 2])
            target_loc = f'stimulus_mod{mod}' 
            self.add_ob(stim_bump, period=stim_period, where=target_loc)

        # Logic
        target_angle = theta
        
        if self.category:
            if 0 < theta <= np.pi:
                target_angle = np.pi/2 # Top
            else:
                target_angle = 3*np.pi/2 # Bottom
        
        if self.anti:
            target_angle = np.mod(target_angle + np.pi, 2*np.pi)
            
        i_target = np.argmin(np.abs(_get_dist(self.theta - target_angle)))
        action_target = i_target + 1 
        
        # GT
        if self.reaction:
             # Response allowed from Stimulus onset
             self.set_gt(action_target, period='stimulus')
        else:
             self.set_gt(action_target, period='decision')
        
        return trial

    def _step(self, action):
        new_trial = False
        reward = 0
        gt = self.gt[self.t_ind]
        
        # Abort logic depends on period
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('delay'):
             if action != 0:
                  new_trial = self.abort
                  reward = self.rewards['abort']
        elif self.in_period('decision') or (self.reaction and self.in_period('stimulus')):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                else:
                    reward = self.rewards['fail']
                    
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


# ---------------------------------------------------------------------
# 2. PoliDM
# ---------------------------------------------------------------------
class PoliDM(PoliEnv):
    def __init__(self, dt=100, anti=False, context=False, multi=False, category=False,
                 modality=1, delay=True, sequential=False, dim_ring=16, cohs=None, sigma=1.0):
        super().__init__(dt=dt, dim_ring=dim_ring)
        self.anti = anti
        self.context = context
        self.multi = multi
        self.category = category
        self.modality = modality 
        self.delay = delay
        self.sequential = sequential
        self.sigma = sigma / np.sqrt(self.dt)
        
        if cohs is None:
            self.cohs = [0.08, 0.16, 0.32]
        else:
            self.cohs = cohs
            
        # Yang19 Timing - NOTE: variable (lambda function) matching task.py
        if self.sequential:
            self.timing = {
                'fixation': lambda: self.rng.uniform(200, 500),
                'sample': lambda: self.rng.choice([200, 400, 600]),
                'delay': lambda: self.rng.choice([200, 400, 800, 1600]), # Yang delay range
                'test': lambda: self.rng.choice([200, 400, 600]),
                'decision': 500 # Yang uses ~500ms post-stimulus response window
            }
        else:
            # Simultaneous DM tasks have NO delay period
            # Yang: Stimulus duration ~400-1200ms (sum of two randoms often)
            # We approximate with uniform or sum of choices.
            self.timing = {
                'fixation': lambda: self.rng.uniform(200, 500),
                'stimulus': lambda: int(self.rng.choice([200, 400, 600]) + self.rng.choice([200, 400, 600])),
                'decision': 500
            }

        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        self.abort = False
        
        self.n_rings = 2 
        name = {'fixation': 0}
        for r in range(self.n_rings):
            name[f'stimulus_mod{r+1}'] = range(1 + r*dim_ring, 1 + (r+1)*dim_ring)
        
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + self.n_rings * dim_ring,), dtype=np.float32, name=name)
        
        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1 + dim_ring, name=name)
        
    def _new_trial(self, **kwargs):
        trial = {'anti': self.anti}
        trial.update(kwargs)
        
        i_theta1 = self.rng.choice(self.choices)
        i_theta2 = (i_theta1 + self.dim_ring//2) % self.dim_ring
        while i_theta2 == i_theta1:
             i_theta2 = self.rng.choice(self.choices)
        theta1 = self.theta[i_theta1]
        theta2 = self.theta[i_theta2]
        
        coh = self.rng.choice(self.cohs)
        sign = self.rng.choice([-1, 1])
        coh1 = 0.5 + sign * coh / 2
        coh2 = 0.5 - sign * coh / 2
        
        i_winner = i_theta1 if coh1 > coh2 else i_theta2
        
        # Category logic: bin winner into top/bottom hemisphere
        if self.category:
            winner_theta = self.theta[i_winner]
            # Top hemisphere (0 < θ <= π) → fixed top target (π/2)
            # Bottom hemisphere (π < θ <= 2π) → fixed bottom target (3π/2)
            if 0 < winner_theta <= np.pi:
                target_angle = np.pi/2
            else:
                target_angle = 3*np.pi/2
            i_target = np.argmin(np.abs(_get_dist(self.theta - target_angle)))
        elif self.anti:
            target_angle = np.mod(self.theta[i_winner] + np.pi, 2*np.pi)
            i_target = np.argmin(np.abs(_get_dist(self.theta - target_angle)))
        else:
            i_target = i_winner
            
        if self.sequential:
            periods = ['fixation', 'sample', 'delay', 'test', 'decision']
            noise_periods = ['sample', 'test']
        else:
            # Simultaneous: NO delay period
            periods = ['fixation', 'stimulus', 'decision']
            noise_periods = ['stimulus']
            
        self.add_period(periods)
        self.add_ob(1, period=periods[:-1], where='fixation')
        
        # Yang19 adds noise to ALL channels (no 'where' parameter)
        self.add_randn(0, self.sigma, noise_periods)

        use_mod1 = (self.modality == 1) or self.multi or self.context
        use_mod2 = (self.modality == 2) or self.multi or self.context
        
        b1 = _gaussianbump(theta1, self.theta, coh1)
        b2 = _gaussianbump(theta2, self.theta, coh2)
        
        if self.sequential:
             # Sequential Logic
             # Rule: 
             # If context=True: 
             #    Signal on self.modality
             #    Distractor on the OTHER modality
             # If context=False:
             #    Signal on active modalities (based on multi/modality params)
             
             if self.context:
                  d1 = _gaussianbump(theta1, self.theta, 0.5)
                  d2 = _gaussianbump(theta2, self.theta, 0.5)
                  
                  if self.modality == 1:
                       # Mod1 = Signal, Mod2 = Distractor
                       self.add_ob(b1, period='sample', where='stimulus_mod1')
                       self.add_ob(b2, period='test', where='stimulus_mod1')
                       self.add_ob(d1, period='sample', where='stimulus_mod2')
                       self.add_ob(d2, period='test', where='stimulus_mod2')
                  else:
                       # Mod2 = Signal, Mod1 = Distractor
                       self.add_ob(d1, period='sample', where='stimulus_mod1')
                       self.add_ob(d2, period='test', where='stimulus_mod1')
                       self.add_ob(b1, period='sample', where='stimulus_mod2')
                       self.add_ob(b2, period='test', where='stimulus_mod2')
             else:
                  # Standard / Multi logic
                  if use_mod1:
                      self.add_ob(b1, period='sample', where='stimulus_mod1')
                      self.add_ob(b2, period='test', where='stimulus_mod1')
                  if use_mod2:
                      self.add_ob(b1, period='sample', where='stimulus_mod2')
                      self.add_ob(b2, period='test', where='stimulus_mod2')
        else:
             # Simultaneous Logic
             if self.context:
                  d = _gaussianbump(theta1, self.theta, 0.5) + _gaussianbump(theta2, self.theta, 0.5)
                  sig = b1 + b2
                  
                  if self.modality == 1:
                       self.add_ob(sig, period='stimulus', where='stimulus_mod1')
                       self.add_ob(d, period='stimulus', where='stimulus_mod2')
                  else:
                       self.add_ob(d, period='stimulus', where='stimulus_mod1')
                       self.add_ob(sig, period='stimulus', where='stimulus_mod2')
             else:
                 if use_mod1: self.add_ob(b1 + b2, period='stimulus', where='stimulus_mod1')
                 if use_mod2: self.add_ob(b1 + b2, period='stimulus', where='stimulus_mod2')
                      
        self.set_gt(i_target + 1, period='decision')
        return trial
        
    def _step(self, action):
        new_trial = False
        reward = 0
        gt = self.gt[self.t_ind]
        if self.in_period('fixation'):
            if action != 0:
                new_trial = True 
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                else:
                    reward = self.rewards['fail']
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


# ---------------------------------------------------------------------
# 3. PoliMatch
# Matches yang19.dms, dmc, etc.
# ---------------------------------------------------------------------
class PoliMatch(PoliEnv):
    def __init__(self, dt=100, matchto='sample', matchgo=True, anti=False, context=False,
                 dim_ring=16, sigma=1.0):
        super().__init__(dt=dt, dim_ring=dim_ring)
        self.matchto = matchto 
        self.matchgo = matchgo 
        self.anti = anti
        self.context = context
        self.sigma = sigma / np.sqrt(self.dt)
        
        # Exact Yang19 Timing (Variable, matching dms_)
        self.timing = {
            'fixation': lambda: self.rng.choice([200, 400, 600]), 
            'sample': lambda: self.rng.choice([200, 400, 600]),
            'delay': lambda: self.rng.choice([200, 400, 800, 1600]),   
            'test': lambda: self.rng.choice([200, 400, 600]), # Second stimulus
            'decision': 500  
        }
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        
        # Yang19 Match uses 2-ring structure (wrapped via _MultiModalityStimulus)
        # 33 dims: 1 fixation + 16 mod1 + 16 mod2
        self.n_rings = 2
        name = {'fixation': 0}
        for r in range(self.n_rings):
            name[f'stimulus_mod{r+1}'] = range(1 + r*dim_ring, 1 + (r+1)*dim_ring)
        
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + self.n_rings * dim_ring,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1 + dim_ring, name=name)
        
        self.half_ring = int(dim_ring/2)

    def _new_trial(self, **kwargs):
        trial = {}
        trial.update(kwargs)
        
        ground_truth = self.rng.choice(['match', 'non-match'])
        i_sample = self.rng.choice(self.dim_ring)
        
        if self.matchto == 'category': # This covers DMC tasks
            sample_cat = (i_sample >= self.half_ring)
            if ground_truth == 'match':
                test_cat = sample_cat
            else:
                test_cat = not sample_cat
            
            if test_cat:
                i_test = self.rng.choice(range(self.half_ring, self.dim_ring))
            else:
                i_test = self.rng.choice(range(0, self.half_ring))
        else:
            if ground_truth == 'match':
                i_test = i_sample
            else:
                i_test = (i_sample + self.half_ring) % self.dim_ring

        sample_theta = self.theta[i_sample]
        test_theta = self.theta[i_test]
        
        stim_sample = _gaussianbump(sample_theta, self.theta, 1)
        stim_test = _gaussianbump(test_theta, self.theta, 1)
        
        self.add_period(['fixation', 'sample', 'delay', 'test', 'decision'])
        
        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision', where='fixation')
        
        # Context mode: sample/test on mod1, distractor on mod2
        if self.context:
            # Generate distractor stimuli
            i_dist_sample = self.rng.choice(self.dim_ring)
            i_dist_test = self.rng.choice(self.dim_ring)
            dist_sample = _gaussianbump(self.theta[i_dist_sample], self.theta, 0.5)
            dist_test = _gaussianbump(self.theta[i_dist_test], self.theta, 0.5)
            
            # Relevant stimuli on mod1, distractors on mod2
            self.add_ob(stim_sample, 'sample', where='stimulus_mod1')
            self.add_ob(stim_test, 'test', where='stimulus_mod1')
            self.add_ob(dist_sample, 'sample', where='stimulus_mod2')
            self.add_ob(dist_test, 'test', where='stimulus_mod2')
        else:
            # Yang19: Randomly place stimulus on mod1 or mod2 (via _MultiModalityStimulus wrapper)
            mod = self.rng.choice([1, 2])
            where_stim = f'stimulus_mod{mod}'
            
            self.add_ob(stim_sample, 'sample', where=where_stim)
            self.add_ob(stim_test, 'test', where=where_stim)
        
        # Yang19 Match: noise ONLY on stimulus channels (not on fixation!)
        # Add noise to both rings since we don't know which will be active
        self.add_randn(0, self.sigma, ['sample', 'test'], where='stimulus_mod1')
        self.add_randn(0, self.sigma, ['sample', 'test'], where='stimulus_mod2')
        
        should_go = False
        if self.matchgo and ground_truth == 'match': should_go = True
        if not self.matchgo and ground_truth == 'non-match': should_go = True
        
        if should_go:
             # Apply anti-mapping if needed (180° rotation)
             target_angle = test_theta
             if self.anti:
                 target_angle = np.mod(test_theta + np.pi, 2*np.pi)
             i_target = np.argmin(np.abs(_get_dist(self.theta - target_angle)))
             self.set_gt(i_target + 1, period='decision')
        else:
             self.set_gt(0, period='decision') 
             
        return trial

    def _step(self, action):
        new_trial = False
        reward = 0
        gt = self.gt[self.t_ind]
        if self.in_period('fixation'):
            if action != 0:
                new_trial = True 
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                else:
                    reward = self.rewards['fail']
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


# ---------------------------------------------------------------------
# Registration Loop
# ---------------------------------------------------------------------
def register_poli_tasks():
    from gym.envs.registration import register
    
    # 1. Reach Family
    register(id='poli.go', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'dt': 100})
    register(id='poli.rtgo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'reaction': True})
    register(id='poli.dlygo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'delay': True})
    register(id='poli.antigo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'anti': True}) # Renamed
    register(id='poli.dlyantigo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'anti': True, 'delay': True}) # Renamed
    register(id='poli.rtantigo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'anti': True, 'reaction': True}) # Renamed
    
    # New Level 1
    register(id='poli.ctxgo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'context': True})
    # (catgo removed)

    # New Level 2 (Compositional)
    register(id='poli.dlyctxgo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'delay': True, 'context': True})
    register(id='poli.antictxgo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'anti': True, 'context': True})
    register(id='poli.rtctxgo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'reaction': True, 'context': True})

    # New Level 3
    register(id='poli.dlyantictxgo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'delay': True, 'anti': True, 'context': True})
    register(id='poli.rtantictxgo', entry_point='curiosity.poli_tasks:PoliReach', kwargs={'reaction': True, 'anti': True, 'context': True})

    # New Level 4
    # (Category removed from Go/DM families for semantic consistency - see validation notes)
    
    # 2. Decision Family
    # Simultaneous
    register(id='poli.dm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'modality': 1, 'delay': False})
    register(id='poli.dm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'modality': 2, 'delay': False})
    register(id='poli.multidm', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'multi': True, 'delay': False})
    register(id='poli.ctxdm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'context': True, 'modality': 1, 'delay': False})
    register(id='poli.ctxdm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'context': True, 'modality': 2, 'delay': False})
    
    # Sequential (Delay)
    register(id='poli.dlydm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'modality': 1, 'sequential': True})
    register(id='poli.dlydm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'modality': 2, 'sequential': True})
    register(id='poli.multidlydm', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'multi': True, 'sequential': True})
    register(id='poli.ctxdlydm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'context': True, 'modality': 1, 'sequential': True})
    register(id='poli.ctxdlydm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'context': True, 'modality': 2, 'sequential': True})
    
    # Missing Level 2 DM
    register(id='poli.antidlydm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'modality': 1, 'sequential': True})
    register(id='poli.antidlydm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'modality': 2, 'sequential': True})
    register(id='poli.antictxdm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'context': True, 'modality': 1, 'delay': False})
    register(id='poli.antictxdm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'context': True, 'modality': 2, 'delay': False})
    register(id='poli.antimultidm', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'multi': True, 'delay': False})

    # Missing Level 3 DM
    register(id='poli.antictxdlydm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'context': True, 'modality': 1, 'sequential': True})
    register(id='poli.antictxdlydm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'context': True, 'modality': 2, 'sequential': True})
    register(id='poli.antimultidlydm', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'multi': True, 'sequential': True})

    # New compositional primitives
    register(id='poli.antidm1', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'modality': 1, 'delay': False})
    register(id='poli.antidm2', entry_point='curiosity.poli_tasks:PoliDM', kwargs={'anti': True, 'modality': 2, 'delay': False})
    
    # 3. Match Family (Renamed: dms->dlyms)
    register(id='poli.dlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': True})
    register(id='poli.dlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': False})
    register(id='poli.catdlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': True})
    register(id='poli.catdlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': False})
    
    # New compositional primitives
    register(id='poli.antidlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': True, 'anti': True})
    register(id='poli.ctxdlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': True, 'context': True})

    # New Level 2 (Compositional Match)
    register(id='poli.antictxdlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': True, 'anti': True, 'context': True})
    register(id='poli.anticatdlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': True, 'anti': True})
    register(id='poli.ctxcatdlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': True, 'context': True})

    # New Level 3 (Compositional Match)
    register(id='poli.antictxcatdlyms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': True, 'anti': True, 'context': True})

    # NMS Variants (Non-Match)
    register(id='poli.antidlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': False, 'anti': True})
    register(id='poli.ctxdlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': False, 'context': True})
    register(id='poli.antictxdlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'sample', 'matchgo': False, 'anti': True, 'context': True})
    register(id='poli.anticatdlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': False, 'anti': True})
    register(id='poli.ctxcatdlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': False, 'context': True})
    register(id='poli.antictxcatdlynms', entry_point='curiosity.poli_tasks:PoliMatch', kwargs={'matchto': 'category', 'matchgo': False, 'anti': True, 'context': True})

    
    print("Registered all poli tasks.")
