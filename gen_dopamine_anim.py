#!/usr/bin/env python3
"""
Organic dopamine vesicle animation for layers/dopamine_block.usda.

Path (SNc-local space, parented to SNc xform):
  SNc soma → nigrostriatal corridor → D1 MSN dendrites → D2 MSN dendrites → D2 soma (hide)

World reference waypoints (match layers/animation.usda pulse-orbs):
  D1 dendrite tips (-4, 12.5, 0)   D2 dendrite tips (4, 12.5, 0)   D2 soma (4, 10, 0)

IMPORTANT: neurons_hq.usda defines DopamineParticles with exactly 20 protoIndices.
Only override positions.timeSamples and invisibleIds.timeSamples.
"""

N        = 20
N_ACTIVE = 10
TOTAL    = 240

# World → SNc-local: world_x=lx, world_y=lz+6, world_z=-ly-10
def world_to_snc(wx, wy, wz):
    return (wx, -(wz + 10.0), wy - 6.0)

# Canonical targets in SNc-local space
D1_DEND = world_to_snc(-4.0, 12.5, 0.0)   # (-4, -10, 6.5)
D1_PROX = world_to_snc(-4.0, 11.0, 0.0)   # descending from D1 arbor
D2_DEND = world_to_snc(4.0, 12.5, 0.0)    # (4, -10, 6.5)
D2_SOMA = world_to_snc(4.0, 10.0, 0.0)    # (4, -10, 4.0) — final hide point

# Per-particle timing — staggered launches, uneven cycle lengths (less robotic)
LAUNCH = [0, 23, 47, 11, 35, 58, 19, 42, 7, 31]
CYCLE  = [52, 58, 61, 55, 63, 57, 60, 54, 62, 59]
HIDE_FRAC = 0.88   # become invisible this fraction through the trip (inside D2 soma)

# 7-waypoint path fractions
WP_T = [0.0, 0.12, 0.32, 0.48, 0.62, 0.78, 1.0]

# Release-site spread at SNc soma
xs_list = [0.10, -0.09, 0.12, -0.07, 0.11, -0.08, 0.13, -0.06, 0.10, -0.10]
# Small offsets inside D2 soma so instances don't stack on one point
d2_jx = [-0.12, 0.10, -0.08, 0.14, -0.10, 0.11, -0.14, 0.08, -0.11, 0.13]
d2_jz = [0.06, -0.05, 0.08, -0.07, 0.05, -0.06, 0.07, -0.04, 0.06, -0.05]

# Arc bulges between waypoints (organic lateral motion)
arc_x = [0.18, -0.14, 0.22, -0.16, 0.12, -0.20, 0.16, -0.11, 0.24, -0.17]
arc_z = [0.35, -0.30, 0.42, -0.38, 0.28, -0.45, 0.38, -0.32, 0.48, -0.36]


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def make_waypoints(i):
    xs = xs_list[i]
    ax = arc_x[i]
    az = arc_z[i]
    d1x, d1y, d1z = D1_DEND
    d2x, d2y, d2z = D2_DEND
    sx, sy, sz = D2_SOMA
    sx += d2_jx[i]
    sz += d2_jz[i]

    return [
        # W0  SNc soma — release
        (xs, -0.90, 0.00),
        # W1  Exiting SNc, ascending toward striatum
        (xs * 0.6 + ax * 0.4, -4.20, 1.10 + az * 0.25),
        # W2  D1 MSN dendritic arbor (left striatum)
        (d1x + xs * 0.08 + ax * 0.15, d1y, d1z + az * 0.20),
        # W3  Proximal D1 — leaving arbor toward midline
        (d1x * 0.85 + ax * 0.10, d1y + 0.35, d1z * 0.72 + az * 0.15),
        # W4  Mid-striatum corridor between D1 and D2
        (ax * 0.35, -9.60, 5.80 + az * 0.25),
        # W5  D2 MSN dendritic arbor (right striatum)
        (d2x + ax * 0.12, d2y, d2z + az * 0.18),
        # W6  Inside D2 MSN soma — absorbed / hidden
        (sx, sy, sz),
    ]


def interp_path(waypoints, p):
    p = max(0.0, min(1.0, p))
    n = len(WP_T) - 1
    for j in range(n):
        t0, t1 = WP_T[j], WP_T[j + 1]
        if p <= t1 or j == n - 1:
            frac = smoothstep((p - t0) / (t1 - t0)) if t1 > t0 else 1.0
            p0, p1 = waypoints[j], waypoints[j + 1]
            return tuple(p0[k] + frac * (p1[k] - p0[k]) for k in range(3))
    return waypoints[-1]


WAYPOINTS = [make_waypoints(i) for i in range(N_ACTIVE)]
PARKED = (0.05, -0.90, 0.05)


def get_pos(i, t):
    if i >= N_ACTIVE:
        return PARKED
    li = LAUNCH[i]
    wps = WAYPOINTS[i]
    cy = CYCLE[i]
    if t <= li:
        return wps[0]
    if t >= li + cy:
        return wps[-1]
    return interp_path(wps, (t - li) / cy)


def is_hidden(i, t):
    """Invisible after entering D2 soma (absorbed). Visible at SNc before/during travel."""
    if i >= N_ACTIVE:
        return True
    li, cy = LAUNCH[i], CYCLE[i]
    hide_at = li + int(cy * HIDE_FRAC)
    if t >= hide_at:
        return True
    return False


# Dense keyframes every 5 frames
keyframes = list(range(0, TOTAL + 1, 5))

pos_lines = ["            point3f[] positions.timeSamples = {"]
for f in keyframes:
    pts = [get_pos(i, f) for i in range(N)]
    formatted = ", ".join(f"({p[0]:.3f}, {p[1]:.4f}, {p[2]:.4f})" for p in pts)
    pos_lines.append(f"                {f}: [{formatted}],")
pos_lines.append("            }")

# Collect all frames where invisible set changes
inv_frames = {0}
for i in range(N_ACTIVE):
    inv_frames.add(LAUNCH[i])
    inv_frames.add(LAUNCH[i] + int(CYCLE[i] * HIDE_FRAC))
    inv_frames.add(LAUNCH[i] + CYCLE[i])
inv_frames = sorted(f for f in inv_frames if f <= TOTAL)


def invisible_at(t):
    ids = list(range(N_ACTIVE, N))
    for i in range(N_ACTIVE):
        if is_hidden(i, t):
            ids.append(i)
    return sorted(ids)


inv_lines = ["            int[] invisibleIds.timeSamples = {"]
prev = None
for f in inv_frames:
    ids = invisible_at(f)
    if ids != prev:
        inv_lines.append(f"                {f}: [{', '.join(str(x) for x in ids)}],")
        prev = ids
inv_lines.append("            }")

output = "\n".join([
    "#usda 1.0",
    "(",
    '    doc = """Organic dopamine vesicle animation — generated by gen_dopamine_anim.py.',
    "    SNc soma → D1 dendrites → D2 dendrites → hide inside D2 MSN soma.",
    "",
    f"    {N} positions per keyframe (protoIndices={N}). {N_ACTIVE} animate, rest parked/invisible.",
    "    Staggered launches and per-particle cycle lengths for organic flow.",
    '    """',
    ")",
    "",
    'over "World"',
    "{",
    '    over "SNc"',
    "    {",
    '        over "DopamineParticles"',
    "        {",
    "            active = true",
    "\n".join(pos_lines),
    "\n".join(inv_lines),
    "        }",
    "    }",
    "}",
])

out_path = "layers/dopamine_block.usda"
with open(out_path, "w") as fh:
    fh.write(output)

print(f"Written to {out_path}")
print(f"  {N_ACTIVE} vesicles | launches {LAUNCH} | cycles {CYCLE}")
print(f"  Path: SNc → D1 dendrites {D1_DEND} → D2 dendrites {D2_DEND} → D2 soma {D2_SOMA}")
labels = ["SNc", "exit", "D1 dendrites", "D1 proximal", "midline", "D2 dendrites", "D2 soma"]
for j, (wp, tf, lbl) in enumerate(zip(WAYPOINTS[0], WP_T, labels)):
    wx, wy, wz = wp[0], wp[2] + 6, -wp[1] - 10
    print(f"  W{j} t={tf:.2f} [{lbl}]: world({wx:+.2f},{wy:+.2f},{wz:+.2f})")
