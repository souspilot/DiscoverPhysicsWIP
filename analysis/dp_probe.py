"""
dp_probe.py — Measure the STRUCTURAL FINGERPRINT of a discovered law by
executing it, not by reading its source.

Design principle: every DiscoverPhysics two-particle world is a STATIC,
POSITION-ONLY central force (the sole exception being 'oscillator', which is
static-but-time-dependent). So we interrogate any submitted law by running it
under controlled perturbations and asking behavioral questions:

  velocity_dependent : hold position fixed, vary initial velocity -> does the
                       induced acceleration change? (Must be FALSE in a static
                       position-only world. Open models frequently violate this
                       by injecting magnetic/cyclotron terms.)
  central            : is the force parallel to the radial line to the source?
  time_dependent     : does the same configuration accelerate differently at
                       different absolute times? (Only 'oscillator' is truly
                       time-dependent.)
  attractive         : does the force point toward the source?
  radial_exponent    : fit |a| ~ r^(-p): recovers the effective power law, and
                       flags single-power-law vs scale-dependent behaviour
                       (the extra_dimensions crossover failure).
  couples_p1, p2     : does the force magnitude respond to p1 / p2?

STATUS TAXONOMY: every law is classified into exactly one of three buckets, so
failures are legible and countable:

  status = "scored"      -> law ran and was structurally fingerprinted
  status = "malformed"   -> law exec'd but every core probe crashed (broken
                            code: e.g. wrong state-vector dimension). Counted as
                            a distinct failure class, NOT silently dropped.
  status = "unsupported" -> law uses a signature this prober does not yet handle
                            (n-body worlds: hubble, dark_matter, three_species,
                            ether, circle). Not a failure -- just out of scope.
"""

import numpy as np
import signal
import contextlib
import io
import inspect
from dataclasses import dataclass
from typing import Optional


# ----------------------------------------------------------------------
# Preloaded modules (avoid re-importing scipy per law -> big speedup)
# ----------------------------------------------------------------------
import numpy as _np
try:
    from scipy.integrate import solve_ivp as _solve_ivp
except Exception:
    _solve_ivp = None


class LawError(Exception):
    pass


# ----------------------------------------------------------------------
# Safe execution with a wall-clock guard (Unix only; SIGALRM)
# ----------------------------------------------------------------------

class _Timeout:
    def __init__(self, seconds=5):
        self.seconds = seconds
        self._supported = hasattr(signal, "SIGALRM")

    def __enter__(self):
        if not self._supported:
            return
        def handler(signum, frame):
            raise LawError("timeout")
        self._old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, *a):
        if not self._supported:
            return
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)


def load_law(src):
    """
    Exec the law source in an isolated namespace and return the callable.
    Injects already-imported numpy / solve_ivp so laws that omit imports still
    run, and laws that import them don't pay the cost twice.
    """
    ns = {"np": _np, "numpy": _np}
    if _solve_ivp is not None:
        ns["solve_ivp"] = _solve_ivp
    try:
        exec(src, ns)
    except Exception as e:
        raise LawError(f"exec failed: {e!r}")
    fn = ns.get("discovered_law")
    if fn is None:
        raise LawError("no discovered_law defined")
    return fn


def law_signature(fn):
    """Return 'two_particle', 'nbody', or 'unknown' from the parameter names."""
    try:
        params = list(inspect.signature(fn).parameters.keys())
    except (ValueError, TypeError):
        return "unknown"
    if "pos1" in params and "pos2" in params:
        return "two_particle"
    if "positions" in params:
        return "nbody"
    return "unknown"


# ----------------------------------------------------------------------
# Acceleration estimate for a two-particle law
# ----------------------------------------------------------------------

def accel_two_particle(fn, pos2, velocity2=(0.0, 0.0), p1=1.0, p2=1.0,
                       pos1=(0.0, 0.0), dt=1e-3):
    """
    Estimate acceleration of particle 2 by running the law for a tiny duration
    and finite-differencing velocity: a ~= (v(dt) - v(0)) / dt.
    Returns np.array([ax, ay]); raises LawError on any failure.
    """
    if _solve_ivp is None:
        raise LawError("scipy unavailable")
    with _Timeout(5), contextlib.redirect_stdout(io.StringIO()):
        try:
            out = fn(list(pos1), list(pos2), p1, p2, list(velocity2), dt)
        except LawError:
            raise
        except Exception as e:
            raise LawError(f"call failed: {e!r}")
    try:
        final_pos, final_vel = out
        vf = np.asarray(final_vel, dtype=float).reshape(-1)[:2]
        if vf.shape[0] < 2:
            raise LawError("returned velocity has <2 components")
    except LawError:
        raise
    except Exception as e:
        raise LawError(f"unexpected return shape: {e!r}")
    v0 = np.asarray(velocity2, dtype=float)
    return (vf - v0) / dt


# ----------------------------------------------------------------------
# Individual structural probes (two-particle worlds)
# ----------------------------------------------------------------------

def _probe_velocity_dependence(fn):
    pos = [4.0, 0.0]
    a0 = accel_two_particle(fn, pos, velocity2=(0.0, 0.0))
    diffs = []
    for v in [(1.0, 0.0), (0.0, 1.0), (-1.5, 0.7)]:
        av = accel_two_particle(fn, pos, velocity2=v)
        diffs.append(np.linalg.norm(av - a0))
    scale = np.linalg.norm(a0) + 1e-9
    return max(diffs) > 0.05 * scale + 1e-6


def _probe_central(fn):
    perp_frac = []
    for pos in [[3.0, 4.0], [-2.0, 5.0], [1.0, -6.0]]:
        a = accel_two_particle(fn, pos, velocity2=(0.0, 0.0))
        r = np.array(pos, dtype=float)
        rn = r / (np.linalg.norm(r) + 1e-12)
        an = np.linalg.norm(a) + 1e-12
        perp = a - np.dot(a, rn) * rn
        perp_frac.append(np.linalg.norm(perp) / an)
    return max(perp_frac) < 0.1


def _probe_attractive(fn):
    a = accel_two_particle(fn, [5.0, 0.0], velocity2=(0.0, 0.0))
    return bool(a[0] < 0)


def _probe_time_dependence(fn):
    """
    Static central law: acceleration depends only on position. We measure accel
    at a fixed position using two different integration windows (dt vs 3*dt from
    rest). For a static law these agree; a law whose rhs references t disagrees.
    Heuristic: can miss weak time dependence, but does not false-positive on
    static laws.
    """
    test_pos = [5.0, 0.0]
    a1 = accel_two_particle(fn, test_pos, velocity2=(0.0, 0.0), dt=1e-3)
    a2 = accel_two_particle(fn, test_pos, velocity2=(0.0, 0.0), dt=3e-3)
    scale = np.linalg.norm(a1) + 1e-9
    return np.linalg.norm(a2 - a1) > 0.1 * scale


def _probe_couples(fn, which):
    base = accel_two_particle(fn, [4.0, 0.0], p1=1.0, p2=1.0)
    if which == "p1":
        pert = accel_two_particle(fn, [4.0, 0.0], p1=3.0, p2=1.0)
    else:
        pert = accel_two_particle(fn, [4.0, 0.0], p1=1.0, p2=3.0)
    b = np.linalg.norm(base) + 1e-12
    return abs(np.linalg.norm(pert) - np.linalg.norm(base)) / b > 0.05


def _probe_force_from_rest(fn):
    """
    Does the law produce ANY acceleration from rest (velocity=0), at a set of
    positions? A purely velocity-dependent force (e.g. magnetic F = qv x B) is
    exactly zero at v=0, so this returns False for such laws. Returns True if a
    static/positional force component exists anywhere in the tested region.
    """
    accels = []
    for r in [1.0, 4.0, 10.0]:
        try:
            a = accel_two_particle(fn, [r, 0.0], velocity2=(0.0, 0.0))
            accels.append(np.linalg.norm(a))
        except LawError:
            continue
    if not accels:
        raise LawError("could not evaluate force from rest at any position")
    # "has a static force" if acceleration from rest is non-negligible somewhere
    return max(accels) > 1e-6


def _fit_exponent(radii, mags):
    """Fit log|a| = c - p*log r over given (radii, mags); return (p, r2)."""
    logr = np.log(np.asarray(radii))
    logm = np.log(np.asarray(mags))
    A = np.vstack([logr, np.ones_like(logr)]).T
    slope, intercept = np.linalg.lstsq(A, logm, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((logm - pred) ** 2)
    ss_tot = np.sum((logm - logm.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return -float(slope), float(r2)


def _probe_radial_exponent(fn):
    """
    Characterize the radial force |a| ~ r^-p by measuring the LOCAL exponent in
    an inner radius band and an outer radius band separately.

    Rationale: a single global log-log fit can score high R^2 on a smooth
    crossover and thus miss scale-dependence. Comparing the inner-band slope to
    the outer-band slope directly measures whether the power law changes with
    scale, which is exactly the extra_dimensions 1/r^2 -> 1/r crossover.

    Returns (exponent_global, exponent_inner, exponent_outer, scale_dependent,
             has_static_force):
      - velocity-only law (no static force from rest):
            (None, None, None, None, False)
      - otherwise the three exponents, a scale_dependent flag set when inner and
        outer exponents differ by more than a tolerance, and True.
    """
    radii = np.geomspace(0.4, 60.0, 20)
    mags, good_r = [], []
    for r in radii:
        try:
            a = accel_two_particle(fn, [float(r), 0.0], velocity2=(0.0, 0.0))
            m = np.linalg.norm(a)
            if np.isfinite(m):
                mags.append(m); good_r.append(float(r))
        except LawError:
            continue
    if len(good_r) < 6:
        raise LawError("too few valid radii to evaluate radial force")

    mags = np.array(mags)
    good_r = np.array(good_r)

    # velocity-only: essentially no static force anywhere
    if np.max(mags) <= 1e-6:
        return None, None, None, None, False

    mask = mags > 1e-12
    if mask.sum() < 6:
        return None, None, None, None, True
    r_pos = good_r[mask]
    m_pos = mags[mask]

    # global exponent (still useful as a summary number)
    exp_global, _ = _fit_exponent(r_pos, m_pos)

    # split into inner and outer bands by radius (log-midpoint split)
    order = np.argsort(r_pos)
    r_sorted = r_pos[order]
    m_sorted = m_pos[order]
    n = len(r_sorted)
    half = n // 2
    # ensure each band has at least 3 points
    if half < 3 or (n - half) < 3:
        # not enough to split; report global only, no scale-dependence claim
        return float(exp_global), None, None, None, True

    exp_inner, r2_inner = _fit_exponent(r_sorted[:half], m_sorted[:half])
    exp_outer, r2_outer = _fit_exponent(r_sorted[half:], m_sorted[half:])

    # scale-dependent if the local exponents differ meaningfully AND each band
    # is itself reasonably power-law (so the difference is a real change of
    # slope, not just noise from a wiggly curve).
    exponent_gap = abs(exp_inner - exp_outer)
    bands_cleanish = (r2_inner > 0.9) and (r2_outer > 0.9)
    scale_dependent = (exponent_gap > 0.3) and bands_cleanish

    return (float(exp_global), float(exp_inner), float(exp_outer),
            bool(scale_dependent), True)


# ----------------------------------------------------------------------
# Fingerprint dataclass and driver
# ----------------------------------------------------------------------

@dataclass
class Fingerprint:
    status: str = "unsupported"     # "scored" | "malformed" | "unsupported"
    signature: str = "unknown"
    velocity_dependent: Optional[bool] = None
    central: Optional[bool] = None
    time_dependent: Optional[bool] = None
    attractive: Optional[bool] = None
    couples_p1: Optional[bool] = None
    couples_p2: Optional[bool] = None
    radial_exponent: Optional[float] = None       # global effective exponent
    radial_exponent_inner: Optional[float] = None  # small-r band exponent
    radial_exponent_outer: Optional[float] = None  # large-r band exponent
    scale_dependent: Optional[bool] = None
    has_static_force: Optional[bool] = None   # False => velocity-only (magnetic) law
    notes: str = ""

    @property
    def executable(self):
        return self.status == "scored"


def fingerprint_two_particle(fn):
    fp = Fingerprint(status="scored", signature="two_particle")
    errors = []

    def safe(f, *a):
        try:
            return f(*a)
        except Exception as e:
            errors.append(repr(e))
            return None

    fp.velocity_dependent = safe(_probe_velocity_dependence, fn)
    fp.central            = safe(_probe_central, fn)
    fp.time_dependent     = safe(_probe_time_dependence, fn)
    fp.attractive         = safe(_probe_attractive, fn)
    fp.couples_p1         = safe(lambda g: _probe_couples(g, "p1"), fn)
    fp.couples_p2         = safe(lambda g: _probe_couples(g, "p2"), fn)

    rad = safe(_probe_radial_exponent, fn)
    if rad is not None:
        (fp.radial_exponent, fp.radial_exponent_inner, fp.radial_exponent_outer,
         fp.scale_dependent, fp.has_static_force) = rad

    core = [fp.velocity_dependent, fp.central, fp.attractive,
            fp.couples_p1, fp.couples_p2]
    if all(v is None for v in core):
        fp.status = "malformed"
        fp.notes = "all probes failed: " + (errors[0] if errors else "unknown")
    elif errors:
        fp.notes = f"{len(errors)} probe(s) failed; first: {errors[0]}"
    return fp


def fingerprint(src):
    """Load a law source and return its structural fingerprint."""
    if not src:
        return Fingerprint(status="malformed", notes="empty law source")
    try:
        fn = load_law(src)
    except LawError as e:
        return Fingerprint(status="malformed", notes=str(e))

    sig = law_signature(fn)
    if sig == "two_particle":
        return fingerprint_two_particle(fn)
    elif sig == "nbody":
        return Fingerprint(status="unsupported", signature="nbody",
                           notes="nbody probing not yet implemented")
    else:
        return Fingerprint(status="unsupported", signature="unknown",
                           notes="unrecognized law signature")


# ----------------------------------------------------------------------
# Per-world ground-truth structural expectations
# ----------------------------------------------------------------------

GROUND_TRUTH = {
    "gravity":          dict(velocity_dependent=False, central=True,  time_dependent=False, attractive=True,  scale_dependent=False),
    "yukawa":           dict(velocity_dependent=False, central=True,  time_dependent=False, attractive=True,  scale_dependent=True),
    "fractional":       dict(velocity_dependent=False, central=True,  time_dependent=False, attractive=True,  scale_dependent=False),
    "coulomb_easy":     dict(velocity_dependent=False, central=True,  time_dependent=False, attractive=True,  scale_dependent=False),
    "extra_dimensions": dict(velocity_dependent=False, central=True,  time_dependent=False, attractive=True,  scale_dependent=True),
    "oscillator":       dict(velocity_dependent=False, central=True,  time_dependent=True,  attractive=None,  scale_dependent=False),
}

TWO_PARTICLE_WORLDS = set(GROUND_TRUTH.keys())


def structural_disagreement(record, fp):
    """
    Compare a fingerprint against the world's ground-truth expectations.
    Returns (n_mismatches, list_of_mismatch_strings), or (None, []) for worlds
    not in GROUND_TRUTH or laws not scored.
    """
    gt = GROUND_TRUTH.get(record.world)
    if gt is None or fp.status != "scored":
        return None, []
    mismatches = []
    for prop, expected in gt.items():
        if expected is None:
            continue
        got = getattr(fp, prop, None)
        if got is None:
            continue
        if bool(got) != bool(expected):
            mismatches.append(f"{prop}: got {got}, expected {expected}")
    return len(mismatches), mismatches


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    try:
        from analysis.dp_parse_json import parse_dir
    except ImportError:
        from dp_parse_json import parse_dir

    d = sys.argv[1] if len(sys.argv) > 1 else "."
    recs = parse_dir(d)

    def s(x):
        return "?" if x is None else ("Y" if x else "n")

    print(f"found {len(recs)} runs in {d}\n")
    hdr = (f"{'world':16s} {'status':10s} {'velDep':6s} {'static':6s} {'centr':5s} "
           f"{'tDep':4s} {'attr':4s} {'p1':2s} {'p2':2s} {'exp':6s} {'scaleDep':8s} "
           f"{'MSE':4s} {'expl':4s} {'struct':7s}")
    print(hdr)
    print("-" * len(hdr))

    n_scored = n_malformed = n_unsupported = 0
    n_veldep_violation = 0
    n_mse_pass_struct_fail = 0

    for r in recs:
        fp = fingerprint(r.final_law_src)
        n_scored      += fp.status == "scored"
        n_malformed   += fp.status == "malformed"
        n_unsupported += fp.status == "unsupported"

        ndis, mism = structural_disagreement(r, fp)
        struct_fail = (ndis is not None and ndis > 0)

        if fp.status == "scored" and r.world in TWO_PARTICLE_WORLDS:
            gt = GROUND_TRUTH[r.world]
            if gt.get("velocity_dependent") is False and fp.velocity_dependent:
                n_veldep_violation += 1
            if r.mse_result == "PASS" and struct_fail:
                n_mse_pass_struct_fail += 1

        if fp.has_static_force is False:
            exp_str = "velOnly"
        elif fp.radial_exponent is None:
            exp_str = ""
        elif fp.scale_dependent and fp.radial_exponent_inner is not None:
            exp_str = f"{fp.radial_exponent_inner:.1f}→{fp.radial_exponent_outer:.1f}"
        else:
            exp_str = f"{fp.radial_exponent:.2f}"
        struct_str = (f"✗×{ndis}" if struct_fail else
                      ("ok" if ndis is not None else "-"))
        print(f"{r.world:16s} {fp.status:10s} {s(fp.velocity_dependent):6s} "
              f"{s(fp.has_static_force):6s} {s(fp.central):5s} {s(fp.time_dependent):4s} "
              f"{s(fp.attractive):4s} {s(fp.couples_p1):2s} {s(fp.couples_p2):2s} "
              f"{exp_str:8s} {s(fp.scale_dependent):8s} {str(r.mse_result):4s} "
              f"{str(r.explanation_score):4s} {struct_str:7s}")
        if struct_fail:
            for m in mism:
                print(f"                 └─ {m}")
        if fp.notes:
            print(f"                 └─ note: {fp.notes}")

    print("\n--- summary ---")
    print(f"scored={n_scored}  malformed={n_malformed}  unsupported(nbody)={n_unsupported}")
    print(f"velocity-dependence violations (static world, law is vel-dependent): {n_veldep_violation}")
    print(f"MSE-pass but structurally-wrong: {n_mse_pass_struct_fail}")