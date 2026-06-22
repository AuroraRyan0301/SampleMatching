from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Dict

import mitsuba as mi

import losses

import math


@dataclass
class OptimizationConfig():
    """Holds configuration options related to a particular optimization run."""
    name: str
    spp: int
    n_iter: int
    importance_iter: int
    importance_iter_stride: int
    importance_grid_resolution: List[int]
    lr: float
    lr_end: float = None

    primal_spp_factor: int = 64
    batch_size: int = None
    lr_schedule: Callable = None
    upsample: List[float] = None
    # Multiplicative factor applied to the learning rate after each upsampling
    # step. The default 0.5 matches prior experiments; e.g. dust-devil uses 0.3.
    upsample_lr_factor: float = 0.5

    base_seed: int = 988378

    render_initial: bool = True
    render_final: bool = True
    preview_stride: int = 100

    checkpoint_initial: bool = True
    checkpoint_final: bool = True
    checkpoint_stride: int = 1000
    # If set, break the optimization loop after reaching this iteration
    # (inclusive).  Upsampling schedule is still driven by n_iter so that the
    # early portion of a run matches a full-length schedule.
    max_iter: int = None
    # If 'zstd', vol files are gzipped via the `zstd` CLI after saving and
    # the .vol is removed (only .vol.zst is kept).
    save_compression: str = None
    # If True and save_compression='zstd', zstd runs in a background thread
    # (max 4 concurrent). save_params returns immediately; an atexit hook
    # drains pending writes before process exit.
    async_compression: bool = False
    # If True, skip saving any param key containing 'emission' (saves disk
    # for scenes that optimize emission like dust-devil / rover).
    skip_emission: bool = False
    # If False, skip writing preview .exr files to disk (still computed for
    # PSNR/loss metrics in-memory). Useful for video runs where preview EXRs
    # would balloon disk usage and aren't needed downstream.
    write_preview_exrs: bool = True
    # 'exr' (default) or 'png' — when 'png', preview images are tonemapped
    # via linear→sRGB and written as 8-bit PNGs (≈3× smaller than EXR).
    preview_format: str = 'exr'
    # Skip writing previews if the destination partition has < threshold_gb
    # free.  Used to gate preview writes when a small partition is being
    # filled by other workloads.
    preview_free_threshold_gb: float = 0.0
    # When False, skip saving the Adam optimizer state (m/v buffers) in
    # checkpoints.  Each opt_state pkl is ~200-500 MB for 256^3+ grids, so
    # disabling this drastically reduces disk usage during reproduction runs
    # where resuming from checkpoint is not needed.
    save_opt_state: bool = True

    preview_spp: int = None
    opt_type: Callable = 'adam'
    opt_args: Dict = None
    loss: Callable = losses.l1

    def __post_init__(self):
        self.upsample_at = set()
        if self.upsample:
            for t in self.upsample:
                assert t >= 0 and t <= 1
                self.upsample_at.add(int(t * self.n_iter))

    def optimizer(self, params):
        opt_type = {'sgd': mi.ad.SGD, 'adam': mi.ad.Adam}[self.opt_type]
        return opt_type(lr=self.lr, params=params, **(self.opt_args or {}))

    def learning_rates(self, scene_config, it_i):
        schedule_factor = 1.0
        if self.lr_schedule not in (None, Schedule.Constant, Schedule.EXPONENTIAL):
            t = it_i / (self.n_iter - 1)
            if self.lr_schedule == Schedule.Last25:
                steps = [0.75, 0.85, 0.95]
            else:
                raise ValueError(f'Unsupported schedule: {self.lr_schedule}')
            for s in steps:
                if t >= s:
                    schedule_factor *= 0.5
        elif self.lr_schedule == Schedule.EXPONENTIAL:
            if self.lr_end is None:
                raise ValueError('lr_end must be specified for exponential schedule')
            t = it_i / (self.n_iter - 1)
            lr_start = self.lr * 1.0
            lr_end = self.lr_end * 1.0
            schedule_factor = math.exp(t * math.log(lr_end / lr_start))
        # after upsampling, scale LRs by the upsampling factor (upsampling is [0.015,0.03,0.05] belike)

        upsampling_factor = 1.0
        if self.upsample:
            for _, upsample_t in enumerate(self.upsample):
                if it_i >= int(upsample_t * self.n_iter):
                    upsampling_factor *= self.upsample_lr_factor

        return {
            k: (schedule_factor * upsampling_factor
                * scene_config.param_lr_factors.get(k, 1.0) * self.lr)
            for k in scene_config.param_keys
        }


    def should_upsample(self, it_i):
        if not self.upsample_at:
            return False
        return it_i in self.upsample_at


class Schedule(IntEnum):
    Constant = 0
    Last25 = 1
    EXPONENTIAL = 2


@dataclass
class IntegratorConfig:
    name: str
    pretty_name: str
    params: Dict

    uses_fd: bool = False
    fd_epsilon: float = None
    fd_spp_multiplier: int = 16

    def __post_init__(self):
        if self.uses_fd:
            assert self.fd_epsilon is not None

    def create(self, **kwargs):
        assert 'max_depth' in kwargs
        d = deepcopy(self.params)
        d.update(kwargs)

        assert d['max_depth'] >= 0
        # TODO: add support for Russian Roulette
        assert 'rr_depth' not in kwargs
        if 'rr_depth' not in self.params:
            d['rr_depth'] = d['max_depth'] + 1000

        return mi.load_dict(d)



_INTEGRATOR_CONFIGS = {}
def add_int_config(name, **kwargs):
    assert name not in _INTEGRATOR_CONFIGS, f'Duplicate integrator config name: {name}'
    _INTEGRATOR_CONFIGS[name] = IntegratorConfig(name, **kwargs)

def get_int_config(name):
    if isinstance(name, IntegratorConfig):
        return deepcopy(name)
    return deepcopy(_INTEGRATOR_CONFIGS[name])


add_int_config(
    'fd-forward',
    pretty_name='Finite differences',
    params={
        'type': 'volpathsimple',
        'use_drt': False,
    },
    uses_fd=True,
    fd_epsilon=5e-3,
)
add_int_config(
    'volpathsimple-drt-quadratic',
    pretty_name='Differential Ratio Tracking (Quatratic)',
    params={
        'type': 'volpathsimple',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': False,
        'use_nee': True,
    },
)
add_int_config(
    'volpathsimple-drt-linear',
    pretty_name='Differential Ratio Tracking (linear)',
    params={
        'type': 'volpathsimple',
        'use_drt': True,
        'use_drt_subsampling': True,
        'use_drt_mis': False,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathsimple-drt-mis',
    pretty_name='Differential Ratio Tracking with mis',
    params={
        'type': 'volpathsimple',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathsimple-drt-mis-n4',
    pretty_name='Differential Ratio Tracking with mis',
    params={
        'type': 'volpathsimple',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathsimple-drt-mis-linear',
    pretty_name='Differential Ratio Tracking with mis (linear)',
    params={
        'type': 'volpathsimple',
        'use_drt': True,
        'use_drt_subsampling': True,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathsimple-drt-n1',
    pretty_name='Differential Ratio Tracking with mis (linear)',
    params={
        'type': 'volpathsimple',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': False,
        'use_nee': True,
        'n_samples_transmittance': 1,
    },
)
add_int_config(
    'volpathsimple-drt-mis-n1',
    pretty_name='Differential Ratio Tracking with mis (linear)',
    params={
        'type': 'volpathsimple',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 1,
    },
)
# ---------------------------------------------------------------------------
# Our method: FM = "forward matching" — along each ray segment, before the next
# scattering event (path vertex), the scattering and transmittance
# extinction-gradient terms are evaluated at shared (matched) sample positions,
# exploiting their negative correlation to cut variance.
#
# *** The paper uses forward matching ONLY *** — the pure matched estimator,
# i.e. the configs WITHOUT `-mis` (also what reproduce.py runs by default).
#
# Config-name suffixes:
#   -sd      same-direction NEE (NEE direction sampled once, shared primal/adjoint)
#   -linear  linear-cost O(n) variant (vs quadratic O(n^2))
#   -nK      K transmittance probes per segment; the paper uses n4, but even n1
#            already beats the DRT baseline at n4, so n1 configs are kept too
#   -mis     OPTIONAL (not used in the paper): MIS-combine the matched estimator
#            with the standard free-flight scattering-gradient estimator; this
#            MIS-combined part is not sample-matched.
#
# Future work: a "post matching" / "post tracking" scheme could also match the
# per-vertex extinction gradient in the -mis case, but the older version of
# Mitsuba 3 that this code builds on has blocking bugs. Email the first author,
# Ruihan Yu, for details.
# ---------------------------------------------------------------------------

# --- quadratic O(n^2), same-direction NEE ---
add_int_config(
    'volpathfm-drt-sd-n4',          # pure matched, no MIS-mix (default pipeline, quadratic)
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_sd',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': False,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathfm-drt-sd-n1',          # pure matched, single transmittance probe
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_sd',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': False,
        'use_nee': True,
        'n_samples_transmittance': 1,
    },
)
add_int_config(
    'volpathfm-drt-mis-sd-n4',      # + MIS-mix with free-flight scattering gradient
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_sd',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathfm-drt-mis-sd-n1',      # + MIS-mix, single transmittance probe
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_sd',
        'use_drt': True,
        'use_drt_subsampling': False,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 1,
    },
)


# --- linear O(n), same-direction NEE ---
add_int_config(
    'volpathfm-linear-drt-sd-n4',       # pure matched, no MIS-mix (default pipeline, linear)
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_linear_sd',
        'use_drt': True,
        'use_drt_subsampling': True,
        'use_drt_mis': False,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathfm-linear-drt-sd-n1',       # pure matched, single transmittance probe
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_linear_sd',
        'use_drt': True,
        'use_drt_subsampling': True,
        'use_drt_mis': False,
        'use_nee': True,
        'n_samples_transmittance': 1,
    },
)
add_int_config(
    'volpathfm-linear-drt-mis-sd-n4',   # + MIS-mix with free-flight scattering gradient
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_linear_sd',
        'use_drt': True,
        'use_drt_subsampling': True,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 4,
    },
)
add_int_config(
    'volpathfm-linear-drt-mis-sd-n1',   # + MIS-mix, single transmittance probe
    pretty_name='Forward Matching (FM)',
    params={
        'type': 'volpathfm_linear_sd',
        'use_drt': True,
        'use_drt_subsampling': True,
        'use_drt_mis': True,
        'use_nee': True,
        'n_samples_transmittance': 1,
    },
)



add_int_config(
    'volpathsimple-basic',
    pretty_name='Free-flight based',
    params={
        'type': 'volpathsimple',
        'use_drt': False,
    },
)

