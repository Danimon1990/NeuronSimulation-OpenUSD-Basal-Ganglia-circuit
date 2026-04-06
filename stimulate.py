"""
stimulate.py — Basal Ganglia Circuit Simulator and USD Stimulus Injector

═══════════════════════════════════════════════════════════
USD LEARNING VALUE OF THIS SCRIPT
═══════════════════════════════════════════════════════════

This script demonstrates four key USD concepts in one pipeline tool:

1. STAGE OPENING AND LAYER STACK INSPECTION
   We open the composed stage (basal_ganglia.usda with all sublayers) and
   then inspect its layer stack to find stimulus.usda specifically.
   This shows how USD exposes the composition architecture programmatically.

2. EDIT TARGET — WRITING TO A SPECIFIC SUBLAYER
   Rather than writing to the composed stage (which would modify the root layer),
   we use stage.SetEditTarget(Usd.EditTarget(stim_layer)) to redirect all
   subsequent edits to stimulus.usda specifically.
   This is the correct way to inject data into a weak sublayer without
   disturbing stronger layers (neurons.usda, animation.usda, etc.).

3. PRIM AND ATTRIBUTE AUTHORING
   We use stage.OverridePrim() (not stage.DefinePrim()) to create 'over' specs.
   An 'over' spec only overrides specific properties; it doesn't redefine the prim.
   This matches the 'over' keyword in USDA.

4. VARIANT SELECTION VIA API
   Optionally setting the displayLOD variant demonstrates how Python code
   can change variant selections on a live stage.

EXAM SUMMARY:
   stage.GetLayerStack()          — returns ordered list of sublayers (strongest first)
   stage.SetEditTarget()          — redirects writes to a specific layer
   stage.OverridePrim(path)       — creates an 'over' spec in the edit target layer
   prim.GetAttribute(name).Set()  — sets an attribute value in the edit target layer
   layer.Save()                   — saves only that one layer file to disk

NEUROSCIENCE ALGORITHM:
   The compute_circuit() function implements a simplified rate-coded model
   of the direct vs indirect pathway competition:

   Direct pathway (Go):   Cortex → D1 MSN → (inhibits) GPi → (releases) Thalamus
   Indirect pathway (NoGo): Cortex → D2 MSN → (inhibits) GPe → (releases) STN → GPi

   Dopamine modulates the competition:
     - Excites D1 MSNs (D1 receptors, Gs): boosts Go
     - Inhibits D2 MSNs (D2 receptors, Gi): suppresses NoGo
   Net effect: dopamine biases toward movement (Go wins more easily).
"""

import argparse
import math
import sys
from pathlib import Path

from pxr import Usd, Sdf, UsdGeom


# ═══════════════════════════════════════════════════════════
# NEUROSCIENCE ALGORITHM
# ═══════════════════════════════════════════════════════════

def compute_circuit(stimulus: float, dopamine_override: float = None) -> dict:
    """
    Compute activation levels for all basal ganglia nuclei given a cortical
    stimulus strength and optional dopamine level override.

    Parameters
    ----------
    stimulus : float
        Cortical stimulus strength (0.0 = no input, 1.0 = maximum input).
        Represents the strength of the corticostriatal signal from Layer V
        pyramidal neurons. Typically 0.0–1.0 but not clamped here.

    dopamine_override : float or None
        If provided, overrides the default tonic dopamine level (0.5).
        Range 0.0–1.0.
          0.0 = Parkinsonian state (90% SNc neuron loss)
          0.5 = Healthy tonic dopamine
          1.0 = Reward burst (phasic dopamine, e.g., unexpected reward)

    Returns
    -------
    dict with keys:
        cortex          float  Cortical activation (= stimulus, 0.0–1.0)
        d1_msn          float  D1 MSN activation (direct/Go pathway, 0.0–1.0)
        d2_msn          float  D2 MSN activation (indirect/NoGo pathway, 0.0–1.0)
        gpi             float  GPi activation (suppressor, high = movement blocked)
        stn             float  STN activation (brake/amplifier, 0.0–1.0)
        thalamus        float  Thalamic relay activation (high = movement released)
        dopamine        float  SNc dopamine level (modulator)
        pathway_winner  str    "direct" (Go wins) or "indirect" (NoGo wins)

    ALGORITHM NOTES (rate-coded model):
        This is not a spiking neuron model. Each value represents a normalized
        firing rate / activation level (0 = silent, 1 = maximally active).

        The key insight of the circuit:
          - Direct pathway REDUCES GPi → INCREASES thalamus
          - Indirect pathway INCREASES GPi → DECREASES thalamus
          - Dopamine AMPLIFIES direct and DAMPENS indirect simultaneously
          - The thalamus threshold (0.5) determines winner

        The dopamine modulation uses additive bias on D1 (excitatory, D1R)
        and subtractive bias on D2 (inhibitory, D2R), which mirrors the
        opposing cAMP effects of D1 (Gs) vs D2 (Gi) receptor signaling.
    """
    # ── Dopamine level ────────────────────────────────────────────────────────
    # Default: 0.5 = healthy tonic dopamine release from SNc
    # Parkinson's: 0.1–0.2 (severe loss of SNc neurons)
    # Reward burst: 0.8–1.0 (phasic dopamine release)
    dopamine = dopamine_override if dopamine_override is not None else 0.5

    # ── Cortex ────────────────────────────────────────────────────────────────
    # Cortical activation directly equals the input stimulus strength.
    # In reality, cortex has its own complex dynamics, but for this circuit
    # model the stimulus IS the cortical output.
    cortex = float(stimulus)

    # ── D1 MSN (Direct pathway — Go) ──────────────────────────────────────────
    # D1 MSN receives:
    #   1. Direct cortical excitation (scaled by cortex)
    #   2. Dopaminergic excitation via D1 receptors (Gs → cAMP → PKA → AMPA facilitation)
    #
    # Formula: d1 = cortex * (1 + dopamine_boost)
    #   dopamine_boost = 0.6 * dopamine  (D1R facilitation factor)
    #   At dopamine=0.5: d1 = cortex * 1.30  (30% boost)
    #   At dopamine=0.0: d1 = cortex * 1.00  (no boost — Parkinsonian)
    #   At dopamine=1.0: d1 = cortex * 1.60  (60% boost — reward burst)
    # Clamp to [0, 1] to represent normalized firing rate
    dopamine_boost_d1 = 0.6 * dopamine  # D1 receptor excitatory modulation
    d1_msn = min(1.0, cortex * (1.0 + dopamine_boost_d1))

    # ── D2 MSN (Indirect pathway — NoGo) ─────────────────────────────────────
    # D2 MSN receives:
    #   1. Direct cortical excitation (scaled by cortex)
    #   2. Dopaminergic INHIBITION via D2 receptors (Gi → ↓cAMP → reduced excitability)
    #
    # Formula: d2 = cortex * (1 - dopamine_suppression)
    #   dopamine_suppression = 0.5 * dopamine  (D2R inhibition factor)
    #   At dopamine=0.5: d2 = cortex * 0.75  (25% suppression)
    #   At dopamine=0.0: d2 = cortex * 1.00  (no suppression — Parkinsonian NoGo overactive)
    #   At dopamine=1.0: d2 = cortex * 0.50  (50% suppression — strong Go bias)
    # Clamp to [0, 1]
    dopamine_suppression_d2 = 0.5 * dopamine  # D2 receptor inhibitory modulation
    d2_msn = max(0.0, cortex * (1.0 - dopamine_suppression_d2))

    # ── GPe (Globus Pallidus externus) ────────────────────────────────────────
    # GPe is inhibited by D2 MSN (GABAergic).
    # GPe baseline activity: 0.8 (tonically active, like GPi)
    # When D2 fires: GPe is suppressed proportionally
    #
    # Formula: gpe = max(0, 0.8 - d2_msn * 0.7)
    #   At d2=0.0: gpe = 0.80  (tonically active, full inhibition of STN)
    #   At d2=0.5: gpe = 0.45  (partial suppression)
    #   At d2=1.0: gpe = 0.10  (nearly silent — STN released)
    gpe = max(0.0, 0.8 - d2_msn * 0.7)

    # ── STN (Subthalamic Nucleus) ─────────────────────────────────────────────
    # STN receives two inputs:
    #   1. Cortical excitation via HYPERDIRECT pathway (direct cortex→STN)
    #   2. Inhibition from GPe (when GPe is active, STN is suppressed)
    #
    # Formula: stn = cortex * 0.4 + (1 - gpe) * 0.6
    #   Hyperdirect term: 0.4 * cortex — the rapid brake signal
    #   Disinhibition term: 0.6 * (1-gpe) — STN activated when GPe is silent
    #   At cortex=1, gpe=0.8: stn = 0.4 + 0.12 = 0.52 (moderate STN activity)
    #   At cortex=1, gpe=0.1: stn = 0.4 + 0.54 = 0.94 (STN highly active when GPe silent)
    stn = min(1.0, cortex * 0.4 + (1.0 - gpe) * 0.6)

    # ── GPi (Globus Pallidus internus) ────────────────────────────────────────
    # GPi is the OUTPUT nucleus. High GPi = movement suppressed (thalamus inhibited).
    # GPi receives:
    #   1. INHIBITION from D1 MSN (direct pathway — reduces GPi)
    #   2. EXCITATION from STN (indirect/hyperdirect pathway — increases GPi)
    # GPi baseline: 0.9 (tonically very active — default state = movement blocked)
    #
    # Formula: gpi = clamp(0.9 - d1_msn * 0.8 + stn * 0.5, 0, 1)
    #   D1 term: -0.8 * d1_msn  — Go pathway suppresses GPi
    #   STN term: +0.5 * stn    — indirect/hyperdirect pathway drives GPi
    #
    # When Go wins:   d1 high, stn low → gpi goes toward 0 → thalamus fires
    # When NoGo wins: d1 low, stn high → gpi stays high → thalamus suppressed
    gpi = max(0.0, min(1.0, 0.9 - d1_msn * 0.8 + stn * 0.5))

    # ── Thalamus ──────────────────────────────────────────────────────────────
    # Thalamus is inhibited by GPi (GABAergic).
    # Thalamic baseline: suppressed (0) — GPi tonic firing keeps it silent.
    # When GPi is suppressed: thalamus can fire.
    #
    # Formula: thalamus = max(0, 1 - gpi * 1.1)
    #   The 1.1 factor means GPi must be reduced to below ~0.91 for any thalamic output.
    #   At gpi=1.0: thalamus = max(0, 1-1.1) = 0.0  (fully suppressed)
    #   At gpi=0.5: thalamus = max(0, 1-0.55) = 0.45 (partial release)
    #   At gpi=0.0: thalamus = 1.0               (fully released — movement!)
    thalamus = max(0.0, 1.0 - gpi * 1.1)

    # ── Pathway winner ────────────────────────────────────────────────────────
    # The thalamus threshold: if thalamus > 0.5, direct pathway wins (action released).
    # This threshold is conventional — a more sophisticated model would use
    # the actual thalamic firing threshold (~15 Hz above rest).
    pathway_winner = "direct" if thalamus > 0.5 else "indirect"

    return {
        "cortex": round(cortex, 4),
        "d1_msn": round(d1_msn, 4),
        "d2_msn": round(d2_msn, 4),
        "gpi": round(gpi, 4),
        "stn": round(stn, 4),
        "thalamus": round(thalamus, 4),
        "dopamine": round(dopamine, 4),
        "pathway_winner": pathway_winner,
    }


# ═══════════════════════════════════════════════════════════
# ASCII DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════

def _bar(value: float, width: int = 10) -> str:
    """
    Return an ASCII progress bar string of the given width.

    Parameters
    ----------
    value : float
        Value between 0.0 and 1.0.
    width : int
        Total number of bar characters.

    Returns
    -------
    str
        A string like '████████░░' for value=0.8, width=10.
        Uses Unicode block character U+2588 (full block) for filled portion
        and U+2591 (light shade) for empty portion.

    Examples
    --------
    _bar(0.0)   → '░░░░░░░░░░'
    _bar(0.5)   → '█████░░░░░'
    _bar(1.0)   → '██████████'
    _bar(0.75)  → '███████░░░'
    """
    filled = int(round(max(0.0, min(1.0, value)) * width))
    empty = width - filled
    return "\u2588" * filled + "\u2591" * empty


def print_circuit_state(values: dict) -> None:
    """
    Print a formatted ASCII visualization of the circuit state.

    Parameters
    ----------
    values : dict
        Dictionary returned by compute_circuit(). Must contain keys:
        cortex, d1_msn, d2_msn, gpi, stn, thalamus, dopamine, pathway_winner.

    Output format:
        A bordered ASCII display showing all nucleus activation levels
        as both a numeric value and a 10-character bar graph.
        The final line announces the pathway winner and thalamic state.
    """
    winner = values["pathway_winner"]
    thal = values["thalamus"]

    direct_marker = "✓" if winner == "direct" else " "
    indirect_marker = "✓" if winner == "indirect" else " "

    thal_state = (
        "FIRES — action released"
        if thal > 0.5
        else "SUPPRESSED — action blocked"
    )

    print("")
    print("  ══════════════════════════════════════════")
    print("    BASAL GANGLIA CIRCUIT SIMULATION")
    print("  ══════════════════════════════════════════")
    print(f"    Stimulus Strength:  {values['cortex']:.2f}")
    print(f"    Dopamine Level:     {values['dopamine']:.2f}")
    print("")
    print(f"    D1 MSN (Go):        {values['d1_msn']:.2f}  {_bar(values['d1_msn'])}")
    print(f"    D2 MSN (NoGo):      {values['d2_msn']:.2f}  {_bar(values['d2_msn'])}")
    print(f"    STN (Brake):        {values['stn']:.2f}  {_bar(values['stn'])}")
    print(f"    GPi (Suppressor):   {values['gpi']:.2f}  {_bar(values['gpi'])}")
    print("")
    print(f"  {direct_marker} DIRECT PATHWAY WINS   (thalamus fires when Go beats NoGo)")
    print(f"  {indirect_marker} INDIRECT PATHWAY WINS (thalamus blocked when NoGo beats Go)")
    print(f"    Thalamus: {thal_state}")
    print("  ══════════════════════════════════════════")
    print("")


# ═══════════════════════════════════════════════════════════
# USD LAYER WRITING
# ═══════════════════════════════════════════════════════════

def write_to_stimulus_layer(values: dict, stage_path: str, lod: str = None) -> None:
    """
    Write computed circuit activation values into layers/stimulus.usda via
    Usd.Stage.SetEditTarget(), keeping all edits confined to that weak sublayer.

    Parameters
    ----------
    values : dict
        Dictionary returned by compute_circuit(). Contains activation levels
        for all circuit nuclei and the pathway_winner string.
    stage_path : str
        Absolute or relative path to basal_ganglia.usda (the root stage).
    lod : str or None
        If provided, sets the displayLOD variant on /World.
        Valid values: 'full_detail', 'circuit_diagram', 'billboard_cards'.

    ═══════════════════════════════════════════════════════════
    EXAM: Sdf vs Usd API for layer authoring
    ═══════════════════════════════════════════════════════════

    Usd.Stage opens the COMPOSED scene (all layers merged).
    Usd.Stage.GetEditTarget() determines which layer receives new opinions.
    Sdf.Layer is the lower-level single-layer interface.

    APPROACH 1: Usd + SetEditTarget (what we use):
      stage = Usd.Stage.Open('basal_ganglia.usda')
      stim_layer = stage.GetLayerStack()[last]  # find stimulus.usda
      stage.SetEditTarget(Usd.EditTarget(stim_layer))
      prim = stage.GetPrimAtPath('/World/Striatum/D1_MSN')
      prim.GetAttribute('neural:activationLevel').Set(0.85)
      stim_layer.Save()  # only saves stimulus.usda, not the whole stage

    APPROACH 2: Sdf direct (alternative, more verbose):
      layer = Sdf.Layer.FindOrOpen('layers/stimulus.usda')
      prim_spec = Sdf.CreatePrimInLayer(layer, '/World/Striatum/D1_MSN')
      prim_spec.specifier = Sdf.SpecifierOver
      attr = Sdf.AttributeSpec(prim_spec, 'neural:activationLevel',
                               Sdf.ValueTypeNames.Float)
      attr.default = 0.85
      layer.Save()

    We use Approach 1 (SetEditTarget) because:
      - It is composition-safe: we open the full stage and navigate by path
      - It is more readable: standard Usd API for attribute setting
      - It is the production-preferred approach in Pixar/Nvidia documentation
      - It lets us optionally set variant selections on the composed stage

    WHY NOT stage.Save() AFTER EDITS?
      stage.Save() would save ALL modified layers, which could include:
        - basal_ganglia.usda (root, might have in-memory edits)
        - neurons.usda (we don't want to touch this)
      Instead, stim_layer.Save() saves ONLY stimulus.usda.
      This is a key discipline: know which layer you're saving.

    UNDERSTANDING stage.OverridePrim():
      prim = stage.OverridePrim('/World/Striatum/D1_MSN')
      This creates an 'over' spec (Sdf.SpecifierOver) at that path in the
      current edit target layer (stimulus.usda). It does NOT:
        - Delete the existing prim definition from neurons.usda
        - Create a 'def' spec that would conflict with the neurons.usda 'def'
      It only adds override opinions to the edit target layer.
      Equivalent USDA: over "D1_MSN" { float neural:activationLevel = 0.85 }
    """
    # ── Find and open the stage ───────────────────────────────────────────────
    root_usda = Path(stage_path)
    if not root_usda.exists():
        print(f"ERROR: Stage file not found: {root_usda}", file=sys.stderr)
        sys.exit(1)

    print(f"  Opening stage: {root_usda}")
    stage = Usd.Stage.Open(str(root_usda))
    if not stage:
        print(f"ERROR: Failed to open stage: {root_usda}", file=sys.stderr)
        sys.exit(1)

    # ── Find stimulus.usda in the layer stack ─────────────────────────────────
    # GetLayerStack(includeSessionLayers=False) returns all sublayers in
    # STRENGTH ORDER: strongest (basal_ganglia.usda) first, weakest last.
    # stimulus.usda is listed last in subLayers, so it should be last in the stack.
    stim_layer = None
    layer_stack = stage.GetLayerStack(includeSessionLayers=False)

    print(f"  Layer stack ({len(layer_stack)} layers):")
    for i, layer in enumerate(layer_stack):
        identifier = layer.identifier
        short_name = Path(identifier).name if identifier else "(anonymous)"
        marker = " <-- target" if short_name == "stimulus.usda" else ""
        print(f"    [{i}] {short_name}{marker}")
        if short_name == "stimulus.usda" or (identifier and identifier.endswith("stimulus.usda")):
            stim_layer = layer

    if stim_layer is None:
        print("ERROR: Could not find stimulus.usda in the layer stack.", file=sys.stderr)
        print("       Ensure basal_ganglia.usda lists @layers/stimulus.usda@ in subLayers.", file=sys.stderr)
        sys.exit(1)

    # ── Set edit target to stimulus.usda ─────────────────────────────────────
    # ALL subsequent prim/attribute edits go to stimulus.usda.
    # Opinions in stronger layers (animation.usda, neurons.usda) are unaffected.
    stage.SetEditTarget(Usd.EditTarget(stim_layer))
    print(f"  Edit target set to: stimulus.usda")

    # ── Author activation levels ──────────────────────────────────────────────
    # Neuron paths → (activation_level, optional_dopamine_influence)
    # We use stage.OverridePrim() which creates 'over' specs in stimulus.usda.
    # Then GetAttribute().Set() authors the attribute value in that spec.

    # GPe is derived here (not stored in values dict) as it's a relay, not a target
    gpe_activation = max(0.0, round(1.0 - values["d2_msn"] * 0.6, 4))

    neuron_writes = [
        ("/World/Cortex/PyramidalNeuron",    values["cortex"],    None),
        ("/World/Striatum/D1_MSN",           values["d1_msn"],    values["dopamine"]),
        ("/World/Striatum/D2_MSN",           values["d2_msn"],    values["dopamine"]),
        ("/World/GPi/GPi_Neuron",            values["gpi"],       None),
        ("/World/GPe/GPe_Neuron",            gpe_activation,      None),
        ("/World/STN/STN_Neuron",            values["stn"],       None),
        ("/World/Thalamus/Thalamic_Neuron",  values["thalamus"],  None),
        ("/World/SNc",                       values["dopamine"],  None),
    ]

    print("")
    print("  Writing neural:activationLevel to stimulus.usda:")

    for prim_path, activation, dopamine_influence in neuron_writes:
        # OverridePrim creates 'over' spec if it doesn't exist, or returns existing
        prim = stage.OverridePrim(prim_path)
        if not prim:
            print(f"  WARNING: Could not override prim at {prim_path}", file=sys.stderr)
            continue

        # Set activation level
        attr = prim.GetAttribute("neural:activationLevel")
        if not attr:
            # Create the attribute if it doesn't exist on the override prim
            attr = prim.CreateAttribute("neural:activationLevel", Sdf.ValueTypeNames.Float)
        attr.Set(float(activation))

        # Optionally set dopamine influence on D1 and D2 MSNs
        if dopamine_influence is not None:
            dopa_attr = prim.GetAttribute("neural:dopamineInfluence")
            if not dopa_attr:
                dopa_attr = prim.CreateAttribute("neural:dopamineInfluence", Sdf.ValueTypeNames.Float)
            dopa_attr.Set(float(dopamine_influence))

        short_path = prim_path.split("/")[-1]
        print(f"    {short_path:<22} neural:activationLevel = {activation:.4f}")

    # ── Optionally set displayLOD variant ────────────────────────────────────
    # EXAM: Variant selection via Python API
    # prim.GetVariantSet(name).SetVariantSelection(value)
    # This writes the variant selection into stimulus.usda (the edit target).
    # Since stimulus.usda is the WEAKEST sublayer, stronger layers can override this.
    # For LOD control, it's better to set this in a stronger layer or the root stage,
    # but we demonstrate it here for completeness.
    if lod is not None:
        world_prim = stage.OverridePrim("/World")
        if world_prim:
            vs = world_prim.GetVariantSet("displayLOD")
            if vs:
                vs.SetVariantSelection(lod)
                print(f"\n  Set displayLOD variant: '{lod}'")
            else:
                print(f"  WARNING: displayLOD variantSet not found on /World", file=sys.stderr)

    # ── Save ONLY stimulus.usda ───────────────────────────────────────────────
    # CRITICAL: We call stim_layer.Save(), NOT stage.Save().
    # stage.Save() would attempt to save ALL dirty layers, including potentially
    # basal_ganglia.usda itself if variant selection changes caused edits there.
    # stim_layer.Save() saves ONLY the stimulus.usda file.
    stim_layer.Save()
    print(f"\n  Saved: {stim_layer.realPath}")


# ═══════════════════════════════════════════════════════════
# PATHWAY OVERRIDE LAYER GENERATION
# ═══════════════════════════════════════════════════════════

# Doc comment written into pathway_override.usda on every run.
# This is the same text as the static comment in the initial template file,
# kept here so the certification explanation survives each ImportFromString().
_OVERRIDE_DOC_LINES = [
    "Pathway Override Layer — written by stimulate.py",
    "",
    "    ═══════════════════════════════════════════════════════════",
    "    USD DATA EXCHANGE PATTERN — Why this is the strongest sublayer",
    "    ═══════════════════════════════════════════════════════════",
    "",
    "    SUBLAYER STACK (strongest first, as listed in basal_ganglia.usda):",
    "      [0] pathway_override.usda  <- THIS FILE (strongest sublayer)",
    "      [1] animation.usda         <- hardcoded Direct-pathway fallback",
    "      [2] connections.usda",
    "      [3] neurons.usda",
    "      [4] shaders.usda",
    "      [5] stimulus.usda          <- weakest (scalar activation data)",
    "",
    "    WHY THIS LAYER MUST BE STRONGER THAN animation.usda:",
    "      animation.usda hard-codes the Direct pathway sequence. Its pulse",
    "      orb xformOp:scale.timeSamples and Soma primvars:displayColor.timeSamples",
    "      always show D1 MSN firing, GPi dimming, and Thalamus releasing.",
    "      This layer sits at position [0] in the subLayers list so its opinions",
    "      WIN over animation.usda in LIVRPS Local opinion resolution.",
    "      Result: when Indirect wins, this layer zeros out the Direct orb",
    "      scales and adds new D2ToGPe / STNToGPi pulse orbs instead.",
    "",
    "    USD DATA EXCHANGE CERTIFICATION PATTERN:",
    "      compute_circuit() produces an activation dict. stimulate.py writes",
    "      that result into TWO sublayers serving different roles:",
    "",
    "        activation scalars -> stimulus.usda           (weakest sublayer)",
    "        pathway animation  -> pathway_override.usda   (strongest sublayer)",
    "",
    "      This file uses the Sdf (single-layer) API for full regeneration:",
    "        Sdf.Layer.FindOrOpen(path)     -- open without a full stage",
    "        layer.ImportFromString(usda)   -- replace entire content atomically",
    "        layer.Save()                   -- write only this file to disk",
    "",
    "      Contrast with stimulus.usda which uses the Usd (composition-aware) API:",
    "        Usd.Stage.Open() + SetEditTarget() + OverridePrim() + attr.Set()",
    "      Both are valid Data Exchange patterns: use Sdf when fully regenerating",
    "      a layer; use Usd + SetEditTarget when patching specific attributes.",
    "",
    "    SUBLAYER OPINION STRENGTH (exam note):",
    "      Sublayer opinions ARE the Local (L) strength in LIVRPS. Within L,",
    "      earlier list position wins. Local beats References, Payloads,",
    "      Inherits, and Specializes. This is why sublayer timeSamples reliably",
    "      override reference or payload defaults without any special API calls.",
    "",
    "    REGENERATION POLICY:",
    "      This file is fully rewritten on each stimulate.py invocation.",
    "      layer.ImportFromString() replaces content atomically, preventing",
    "      stale opinions from previous runs accumulating in the layer.",
]


def _soma_color_lines(path_parts: list, keyframes: list) -> list:
    """
    Build nested 'over' USDA lines for a Soma primvars:displayColor.timeSamples.

    Parameters
    ----------
    path_parts : list of str
        Prim names from World downward, e.g. ['Striatum', 'D1_MSN', 'Soma'].
    keyframes : list of (frame, r, g, b)
        Animation keyframes. frame is int, r/g/b are float 0.0-1.0.

    Returns
    -------
    list of str
        USDA lines with 4-space base indent (assumes inside 'over "World"').

    EXAM NOTE:
        These 'over' specs live in pathway_override.usda (a sublayer).
        Sublayer opinions are Local (L) in LIVRPS — the strongest possible.
        They override the referenced geometry's displayColor (which comes
        from R in LIVRPS) without needing any special API or arc override.
    """
    lines = []
    indent = "    "
    for part in path_parts:
        lines.append(f"{indent}over \"{part}\"")
        lines.append(f"{indent}{{")
        indent += "    "

    lines.append(f"{indent}color3f[] primvars:displayColor.timeSamples = {{")
    for frame, r, g, b in keyframes:
        lines.append(f"{indent}    {frame}: [({r:.4f}, {g:.4f}, {b:.4f})],")
    lines.append(f"{indent}}}")

    for _ in path_parts:
        indent = indent[:-4]
        lines.append(f"{indent}}}")

    return lines


def _snc_dopamine_lines(dopamine: float) -> list:
    """
    USDA lines for SNc Soma displayColor.timeSamples driven by dopamine level.

    The soma pulses yellow-green with brightness and pulse amplitude scaled
    by the dopamine value: 0.1 (Parkinsonian) = dim, infrequent; 0.5
    (healthy) = moderate; 1.0 (reward burst) = very bright, strong pulse.

    This is the visual hook that makes --dopamine actually visible in the scene.
    The static SNcDopamineLight in basal_ganglia.usda has a fixed intensity
    schedule; only this layer, regenerated per run, responds to the parameter.

    EXAM NOTE:
        SNc uses a payload arc (P in LIVRPS). Local sublayer opinions (L)
        beat payload defaults. pathway_override.usda is L, so its Soma
        displayColor.timeSamples override the asset's static color.
        The dopamine value is authored as data in stimulus.usda and as
        visual animation here — demonstrating the two-layer Data Exchange split.
    """
    # dim baseline color — scales with dopamine so Parkinsonian state looks dull
    dim_r = round(max(0.2, 0.2 + dopamine * 0.55), 3)
    dim_g = round(max(0.3, 0.3 + dopamine * 0.65), 3)
    dim_b = 0.02

    # peak pulse color — brighter the higher the dopamine
    peak_r = round(min(1.0, dim_r + 0.35), 3)
    peak_g = round(min(1.0, dim_g + 0.30), 3)
    peak_b = 0.06

    # Three pulses over 240 frames (every 80 frames).
    # pulse_half_width: dopamine=1.0 → 25 frames; dopamine=0.1 → 8 frames
    pw = max(6, int(dopamine * 25))
    keyframes = [
        (0,       dim_r, dim_g, dim_b),
        (pw,      peak_r, peak_g, peak_b),
        (pw * 2,  dim_r, dim_g, dim_b),
        (80,      dim_r, dim_g, dim_b),
        (80 + pw, peak_r, peak_g, peak_b),
        (80 + pw * 2, dim_r, dim_g, dim_b),
        (160,     dim_r, dim_g, dim_b),
        (160 + pw, peak_r, peak_g, peak_b),
        (160 + pw * 2, dim_r, dim_g, dim_b),
        (240,     dim_r, dim_g, dim_b),
    ]

    return _soma_color_lines(["SNc", "Soma"], keyframes)


def _direct_pathway_lines(values: dict) -> list:
    """
    USDA body lines for a Direct (Go) pathway win.

    Visual story:
      D1 MSN flashes bright blue (firing, suppressing GPi).
      GPi darkens (being suppressed — the gate opens).
      Thalamus flashes gold (released! action output propagating to cortex).
      D2 MSN, GPe, STN stay at resting colors (they lost the competition).
      SNc Soma pulses with brightness driven by values['dopamine'].

    The Direct-pathway pulse orbs (D1ToGPi, GPiToThalamus) defined in
    animation.usda are NOT touched here — this layer stays silent on them,
    so animation.usda wins for those prims and the Direct orbs play normally.
    """
    lines = [
        "    # Direct pathway win: D1 MSN fires, GPi suppressed, Thalamus released.",
        "    # Soma displayColor.timeSamples flash winning nuclei.",
        "    # Direct pulse orbs (D1ToGPi, GPiToThalamus) are left to animation.usda.",
    ]

    # D1 MSN: blue flash — firing hard, suppressing GPi
    lines += _soma_color_lines(
        ["Striatum", "D1_MSN", "Soma"],
        [(55, 0.2, 0.4, 1.0), (65, 0.5, 0.7, 1.0), (80, 0.3, 0.5, 1.0), (90, 0.2, 0.4, 1.0)],
    )

    # GPi: darkens as D1 MSN GABA suppresses it (the key visual moment)
    lines += _soma_color_lines(
        ["GPi", "GPi_Neuron", "Soma"],
        [(110, 0.75, 0.6, 0.9), (125, 0.2, 0.1, 0.3), (145, 0.03, 0.01, 0.06), (180, 0.75, 0.6, 0.9)],
    )

    # Thalamus: gold flash — released from GPi suppression, action output!
    lines += _soma_color_lines(
        ["Thalamus", "Thalamic_Neuron", "Soma"],
        [(140, 0.75, 0.6, 0.9), (152, 1.0, 0.92, 0.6), (168, 0.9, 0.8, 0.5), (195, 0.75, 0.6, 0.9)],
    )

    # SNc Soma: dopamine-driven pulse — brightness reflects --dopamine value
    lines += _snc_dopamine_lines(values["dopamine"])

    return lines


def _indirect_pathway_lines(values: dict) -> list:
    """
    USDA body lines for an Indirect (NoGo) pathway win.

    Visual story:
      CortexToStriatum orb redirected to D2 MSN (4,10,0) instead of D1 (-4,10,0).
      D2 MSN flashes amber (firing, suppressing GPe).
      GPe darkens (suppressed by D2 GABA — releases STN from inhibition).
      STN flashes teal (disinhibited, now driving GPi).
      GPi stays bright purple (active — movement blocked).
      Thalamus stays dark throughout (suppressed, no action output).
      Direct-pathway pulse orbs are zeroed so they don't play.
      New Indirect orbs (D2ToGPe, STNToGPi) are defined in this layer.
      SNc Soma pulses with brightness driven by values['dopamine'].
    """
    lines = [
        "    # Indirect pathway win: D2 MSN fires, GPe suppressed, STN active,",
        "    # GPi stays HIGH (movement blocked), Thalamus stays dark.",
    ]

    # Zero out the Direct-pathway pulse orbs from animation.usda.
    # Because this layer is stronger, scale=(0,0,0) timeSamples win.
    lines += [
        "",
        "    # Suppress Direct-pathway pulse orbs and redirect CortexToStriatum.",
        "    # This layer's timeSamples beat animation.usda's values (stronger sublayer).",
        "    over \"PulseOrbs\"",
        "    {",
        "        # Redirect the corticostriatal orb to D2 MSN (4,10,0) instead of",
        "        # D1 MSN (-4,10,0). animation.usda hardcodes the D1 destination;",
        "        # this stronger-layer override changes the end point for NoGo wins.",
        "        # D2 MSN world pos: Striatum group (0,10,0) + local offset (+4,0,0).",
        "        over \"CortexToStriatum\"",
        "        {",
        "            double3 xformOp:translate.timeSamples = {",
        "                0:  (0.0, 18.0, 0.0),",
        "                20: (0.0, 18.0, 0.0),",
        "                40: (2.0, 14.0, 0.0),",
        "                60: (4.0, 10.0, 0.0)",
        "            }",
        "            color3f[] primvars:displayColor = [(1.0, 0.65, 0.15)]",
        "        }",
        "        over \"D1ToGPi\"",
        "        {",
        "            double3 xformOp:scale.timeSamples = {",
        "                0: (0.0, 0.0, 0.0),",
        "                240: (0.0, 0.0, 0.0)",
        "            }",
        "        }",
        "        over \"GPiToThalamus\"",
        "        {",
        "            double3 xformOp:scale.timeSamples = {",
        "                0: (0.0, 0.0, 0.0),",
        "                240: (0.0, 0.0, 0.0)",
        "            }",
        "        }",
        "",
        "        # Indirect-pathway pulse orbs defined as new prims in this layer.",
        "        # 'def' inside an 'over' container is valid — it adds new children",
        "        # to the composed PulseOrbs scope without redefining the scope itself.",
        "        def Sphere \"D2ToGPe\"",
        "        {",
        "            double radius = 0.2",
        "            color3f[] primvars:displayColor = [(1.0, 0.7, 0.2)]",
        "            double3 xformOp:translate.timeSamples = {",
        "                80: (4.0, 10.0, 0.0),",
        "                95: (4.5, 6.0, 0.0),",
        "                110: (5.0, 2.0, 0.0)",
        "            }",
        "            double3 xformOp:scale.timeSamples = {",
        "                79: (0.0, 0.0, 0.0),",
        "                80: (1.0, 1.0, 1.0),",
        "                109: (1.0, 1.0, 1.0),",
        "                110: (0.0, 0.0, 0.0)",
        "            }",
        "            uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]",
        "        }",
        "",
        "        def Sphere \"STNToGPi\"",
        "        {",
        "            double radius = 0.2",
        "            color3f[] primvars:displayColor = [(0.0, 0.85, 0.85)]",
        "            double3 xformOp:translate.timeSamples = {",
        "                120: (8.0, -4.0, 2.0),",
        "                140: (1.5, -1.0, 1.0),",
        "                160: (-5.0, 2.0, 0.0)",
        "            }",
        "            double3 xformOp:scale.timeSamples = {",
        "                119: (0.0, 0.0, 0.0),",
        "                120: (1.0, 1.0, 1.0),",
        "                159: (1.0, 1.0, 1.0),",
        "                160: (0.0, 0.0, 0.0)",
        "            }",
        "            uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]",
        "        }",
        "    }",
    ]

    # D2 MSN: amber flash — firing, suppressing GPe
    lines += _soma_color_lines(
        ["Striatum", "D2_MSN", "Soma"],
        [(55, 1.0, 0.6, 0.1), (65, 1.0, 0.85, 0.2), (80, 1.0, 0.7, 0.15), (90, 1.0, 0.6, 0.1)],
    )

    # GPe: darkens as D2 MSN GABA suppresses it (releases STN from inhibition)
    lines += _soma_color_lines(
        ["GPe", "GPe_Neuron", "Soma"],
        [(85, 0.75, 0.6, 0.9), (100, 0.2, 0.1, 0.3), (118, 0.03, 0.01, 0.06), (155, 0.75, 0.6, 0.9)],
    )

    # STN: teal flash — disinhibited (GPe no longer holds it back)
    lines += _soma_color_lines(
        ["STN", "STN_Neuron", "Soma"],
        [(100, 0.75, 0.6, 0.9), (118, 0.0, 0.85, 0.85), (135, 0.0, 0.65, 0.65), (165, 0.75, 0.6, 0.9)],
    )

    # GPi: stays bright purple — STN is driving it, movement stays blocked
    lines += _soma_color_lines(
        ["GPi", "GPi_Neuron", "Soma"],
        [(120, 0.6, 0.3, 0.85), (160, 0.6, 0.3, 0.85), (200, 0.75, 0.6, 0.9)],
    )

    # Thalamus: stays dark throughout — GPi never releases it
    lines += _soma_color_lines(
        ["Thalamus", "Thalamic_Neuron", "Soma"],
        [(0, 0.3, 0.2, 0.4), (240, 0.3, 0.2, 0.4)],
    )

    # SNc Soma: dopamine-driven pulse — brightness reflects --dopamine value
    lines += _snc_dopamine_lines(values["dopamine"])

    return lines


def _build_pathway_override_usda(values: dict) -> str:
    """
    Generate the complete USDA text for pathway_override.usda.

    The output is a fully self-contained sublayer that:
      1. Contains the certification doc comment (regenerated each run so it
         survives ImportFromString replacing the layer content).
      2. Writes pathway-specific Soma displayColor.timeSamples to flash
         the neurons that belong to the winning pathway.
      3. For Indirect wins: zeros out Direct pulse orb scales and defines
         new D2ToGPe / STNToGPi Sphere prims.
      4. For Direct wins: stays silent on Direct orbs so animation.usda
         plays them unmodified.

    Parameters
    ----------
    values : dict
        Output from compute_circuit(). Uses 'pathway_winner' to branch.

    Returns
    -------
    str
        Valid USDA text for Sdf.Layer.ImportFromString().
    """
    dq = '"""'
    pathway = values["pathway_winner"]

    # Build the header with the certification doc comment
    doc_body = "\n".join(_OVERRIDE_DOC_LINES)
    lines = [
        "#usda 1.0",
        "(",
        f"    doc = {dq}",
        doc_body,
        f"    {dq}",
        '    defaultPrim = "World"',
        ")",
        "",
        'over "World"',
        "{",
    ]

    if pathway == "direct":
        lines += _direct_pathway_lines(values)
    else:
        lines += _indirect_pathway_lines(values)

    lines += ["}", ""]
    return "\n".join(lines)


def write_to_pathway_override_layer(values: dict, stage_path: str) -> None:
    """
    Regenerate layers/pathway_override.usda with pathway-specific animation.

    Completely replaces the layer content on each run using the Sdf
    single-layer API (layer.ImportFromString), ensuring no stale opinions
    remain from previous invocations.

    Parameters
    ----------
    values : dict
        Output from compute_circuit(). 'pathway_winner' selects the branch.
    stage_path : str
        Path to basal_ganglia.usda (used to locate pathway_override.usda).

    USD EXAM — Sdf vs Usd API choice:
        This function uses Sdf.Layer directly rather than Usd.Stage + SetEditTarget.
        Reason: we are fully regenerating the layer from scratch on every call.
        ImportFromString() is the correct tool for atomic full-layer replacement —
        it parses and installs the new content in one operation, with no risk of
        stale specs from a previous pathway surviving the update.

        Contrast with write_to_stimulus_layer() which patches only specific
        attribute values using SetEditTarget. That function keeps the layer
        structure intact and only changes the neural:activationLevel opinions
        — the right choice when only a few values need updating.

        Rule of thumb:
          Full layer regeneration  → Sdf.Layer + ImportFromString
          Targeted attribute patch → Usd.Stage + SetEditTarget + attr.Set
    """
    override_path = Path(stage_path).parent / "layers" / "pathway_override.usda"
    if not override_path.exists():
        print(
            f"ERROR: pathway_override.usda not found at {override_path}\n"
            f"       Create the file (or run from the BG_Circuit directory).",
            file=sys.stderr,
        )
        sys.exit(1)

    pathway = values["pathway_winner"]
    print(f"\n  Generating pathway_override.usda  (pathway_winner = '{pathway}')")

    usda = _build_pathway_override_usda(values)

    # Sdf.Layer.FindOrOpen() checks the USD layer registry first (the layer
    # may already be cached from write_to_stimulus_layer opening the stage),
    # then falls back to opening from disk. Either way we get a live SdfLayer.
    layer = Sdf.Layer.FindOrOpen(str(override_path))
    if layer is None:
        print(f"ERROR: Sdf could not open pathway_override.usda", file=sys.stderr)
        sys.exit(1)

    # ImportFromString() replaces the entire layer content atomically.
    # This is the Sdf-level equivalent of 'delete and recreate' — no stale
    # opinions survive. The method returns False if the USDA fails to parse.
    if not layer.ImportFromString(usda):
        print(
            "ERROR: Generated USDA failed to parse.\n"
            "       Check _build_pathway_override_usda() for syntax errors.",
            file=sys.stderr,
        )
        sys.exit(1)

    layer.Save()
    print(f"  Saved: {layer.realPath}")


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main() -> None:
    """
    Command-line interface for the basal ganglia circuit simulator.

    Usage examples
    --------------
    # Default: healthy dopamine (0.5), moderate stimulus (0.75)
    python stimulate.py

    # Parkinsonian state: low dopamine, any stimulus strength
    python stimulate.py --strength 0.8 --dopamine 0.1

    # Reward burst: high dopamine
    python stimulate.py --strength 0.6 --dopamine 0.95

    # Dry run: see circuit state without writing USD
    python stimulate.py --strength 0.9 --dry-run

    # Switch to circuit diagram LOD while writing activation values
    python stimulate.py --strength 0.75 --lod circuit_diagram

    # Sweep stimulus strength to see tipping point
    for s in 0.2 0.4 0.6 0.8 1.0; do
        python stimulate.py --strength $s --dry-run
    done
    """
    parser = argparse.ArgumentParser(
        prog="stimulate.py",
        description=(
            "Compute basal ganglia circuit activations and inject into USD stimulus layer.\n"
            "\n"
            "USD EXAM DEMO: This script shows how to write to a specific sublayer\n"
            "using Usd.Stage.SetEditTarget() + Usd.EditTarget(layer).\n"
            "All writes go to layers/stimulus.usda — the weakest sublayer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        metavar="FLOAT",
        help=(
            "Cortical stimulus strength 0.0–1.0. "
            "0.0 = no cortical input, 1.0 = maximum drive. "
            "Default: 0.75 (moderate stimulus, Go pathway typically wins)."
        ),
    )

    parser.add_argument(
        "--dopamine",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Override dopamine level 0.0–1.0. "
            "0.0 = Parkinsonian (no dopamine), 0.5 = healthy tonic (default), "
            "1.0 = reward burst. If omitted, uses healthy tonic 0.5."
        ),
    )

    parser.add_argument(
        "--lod",
        choices=["full_detail", "circuit_diagram", "billboard_cards"],
        default=None,
        help=(
            "Set the displayLOD variant on /World in stimulus.usda. "
            "full_detail = full geometry (default state), "
            "circuit_diagram = bounding boxes only, "
            "billboard_cards = textured quad sprites."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=False,
        help=(
            "Print circuit state to stdout but do NOT write to stimulus.usda. "
            "Useful for previewing values before committing to the USD scene."
        ),
    )

    args = parser.parse_args()

    # Validate input range
    if not (0.0 <= args.strength <= 1.0):
        print(f"WARNING: --strength {args.strength} is outside [0.0, 1.0]. Clamping.", file=sys.stderr)
        args.strength = max(0.0, min(1.0, args.strength))

    if args.dopamine is not None and not (0.0 <= args.dopamine <= 1.0):
        print(f"WARNING: --dopamine {args.dopamine} is outside [0.0, 1.0]. Clamping.", file=sys.stderr)
        args.dopamine = max(0.0, min(1.0, args.dopamine))

    # ── Compute circuit state ─────────────────────────────────────────────────
    values = compute_circuit(
        stimulus=args.strength,
        dopamine_override=args.dopamine,
    )

    # ── Print circuit state ───────────────────────────────────────────────────
    print_circuit_state(values)

    # ── Write to USD (unless --dry-run) ───────────────────────────────────────
    if args.dry_run:
        print("  (dry-run: USD not modified)")
        return

    # Locate basal_ganglia.usda relative to this script's directory
    script_dir = Path(__file__).parent
    stage_path = script_dir / "basal_ganglia.usda"

    if not stage_path.exists():
        print(
            f"ERROR: basal_ganglia.usda not found at {stage_path}\n"
            f"       Run this script from its directory or ensure the USD project is built.",
            file=sys.stderr,
        )
        sys.exit(1)

    write_to_stimulus_layer(
        values=values,
        stage_path=str(stage_path),
        lod=args.lod,
    )

    write_to_pathway_override_layer(
        values=values,
        stage_path=str(stage_path),
    )

    print("  Done — stimulus.usda and pathway_override.usda updated.")


if __name__ == "__main__":
    main()
