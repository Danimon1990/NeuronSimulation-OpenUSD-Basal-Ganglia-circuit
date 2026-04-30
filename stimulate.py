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
import random
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


class _OverTree:
    """
    Merges displayColor.timeSamples overrides by shared prim path, then renders
    them as properly-nested USDA 'over' blocks where each prim appears exactly once.

    Without this, calling _soma_color_lines twice for paths that share a prefix
    (e.g. ['Striatum','D1_MSN','Soma'] and ['Striatum','D1_MSN','Dendrites','Primary_00'])
    produces two 'over "Striatum"' blocks in the same layer — a parse error.

    Usage:
        t = _OverTree()
        t.add(["Cortex", "PyramidalNeuron", "Soma"], kf)
        t.add(["Cortex", "PyramidalNeuron", "Axon", "MainAxon"], kf2)
        lines += t.to_lines()   # produces ONE over "Cortex" { over "PyramidalNeuron" { ... } }
    """

    def __init__(self):
        self._kf = None       # keyframes if this node is a color leaf
        self._children = {}   # child name → _OverTree

    def add(self, path: list, kf: list):
        """Register a (path, keyframes) override into the tree."""
        if not path:
            self._kf = kf
            return
        name = path[0]
        if name not in self._children:
            self._children[name] = _OverTree()
        self._children[name].add(path[1:], kf)

    def _render_body(self, indent: str) -> list:
        """Render the contents of this node (color attr + child over blocks)."""
        lines = []
        if self._kf is not None:
            lines.append(f'{indent}color3f[] primvars:displayColor.timeSamples = {{')
            for frame, r, g, b in self._kf:
                lines.append(f'{indent}    {frame}: [({r:.4f}, {g:.4f}, {b:.4f})],')
            lines.append(f'{indent}}}')
        for name, child in self._children.items():
            lines += [f'{indent}over "{name}"', f'{indent}{{']
            lines += child._render_body(indent + "    ")
            lines += [f'{indent}}}']
        return lines

    def to_lines(self, indent: str = "    ") -> list:
        """Render top-level children as over blocks (for insertion inside over "World")."""
        lines = []
        for name, child in self._children.items():
            lines += [f'{indent}over "{name}"', f'{indent}{{']
            lines += child._render_body(indent + "    ")
            lines += [f'{indent}}}']
        return lines


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


# ─── Anatomy prim name sets ───────────────────────────────────────────────────
# Each neuron type has different dendrite names matching their asset files.
_MSN_DENDRITE_NAMES       = ["Primary_00", "Primary_01", "Primary_02",
                              "Primary_03", "Primary_04", "Primary_05"]
_GENERIC_DENDRITE_NAMES   = ["Primary_00", "Primary_01", "Primary_02", "Primary_03"]
_PYRAMIDAL_BASAL_NAMES    = ["Basal_00", "Basal_01", "Basal_02", "Basal_03"]


def _glow_many(base_path: list, child_names: list, keyframes: list) -> list:
    """
    Animate primvars:displayColor.timeSamples on multiple sibling prims with
    identical keyframes — used to light all dendrites of a neuron simultaneously.

    base_path   : path from World to the scope, e.g. ['Striatum', 'D1_MSN', 'Dendrites']
    child_names : e.g. ['Primary_00', 'Primary_01', ...]
    keyframes   : list of (frame, r, g, b)
    """
    lines = []
    for name in child_names:
        lines += _soma_color_lines(base_path + [name], keyframes)
    return lines


def _nc(path: list, kf: list, base: str) -> list:
    """
    Nested Color: generate over blocks + displayColor.timeSamples starting at an
    already-open indentation level (base). Use this INSIDE an already-opened over
    block to avoid the duplicate-prim error that happens when _soma_color_lines is
    called multiple times for children of the same parent prim.

    path : prim path relative to base, e.g. ['Soma'] or ['Dendrites', 'Primary_00']
    kf   : [(frame, r, g, b), ...]
    base : starting indent string (e.g. '            ' = 3 levels = 12 spaces)
    """
    lines = []
    ind = base
    for part in path:
        lines += [f'{ind}over "{part}"', f'{ind}{{']
        ind += "    "
    lines.append(f'{ind}color3f[] primvars:displayColor.timeSamples = {{')
    for frame, r, g, b in kf:
        lines.append(f'{ind}    {frame}: [({r:.4f}, {g:.4f}, {b:.4f})],')
    lines.append(f'{ind}}}')
    for _ in path:
        ind = ind[:-4]
        lines.append(f'{ind}}}')
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


def _dopamine_particles_lines(dopamine: float) -> list:
    """
    USDA lines overriding SNc/DopamineParticles positions.timeSamples so that
    vesicle density and travel speed scale with the dopamine level.

    SNc world position: (0, 6, -10).  PointInstancer positions are LOCAL to SNc.
    Target positions in SNc-local space (derived from world targets minus SNc origin):
      D1 MSN  (-4, 10, 0) - (0, 6, -10) = (-4,  4, 10)
      D2 MSN  ( 4, 10, 0) - (0, 6, -10) = ( 4,  4, 10)

    Conveyor-belt model (same trick as the original animation.usda):
      - N_active particles are pre-distributed along the two paths at t=0.
      - Every T frames they advance one "slot" forward along the path.
      - The leading particle arrives at the MSN soma; at frame T+1 the array
        hard-resets to the frame-0 distribution (disguised by particle density).
      - N_active and T both scale with dopamine:
          dopamine=0.0  → 0 active, 20 resting near soma, no movement (Parkinsonian)
          dopamine=0.5  → 10 active (5 D1, 5 D2), T≈78 frames
          dopamine=1.0  → 20 active (10 D1, 10 D2), T=30 frames (fast burst)

    This override lives in pathway_override.usda (strongest sublayer) so it beats
    the static Y-only motion in animation.usda, which never reaches the MSN targets.

    EXAM NOTE:
        PointInstancer.positions is a point3f[] attribute.
        All N instance positions are encoded as one array per keyframe.
        We only override positions here; protoIndices (which prototype mesh each
        instance uses) and the Vesicle prototype radius remain from the asset /
        animation.usda overrides, untouched by this layer.
    """
    rng = random.Random(42)  # fixed seed → reproducible output across runs

    # SNc-local space targets
    D1_LOCAL = (-4.0,  4.0, 10.0)   # world (-4, 10, 0)
    D2_LOCAL = ( 4.0,  4.0, 10.0)   # world ( 4, 10, 0)
    SOMA     = ( 0.0, -0.5,  0.0)   # near SNc soma centre in local space

    # Particle budget
    n_active = round(dopamine * 20)
    n_d1     = n_active // 2
    n_d2     = n_active - n_d1
    n_rest   = 20 - n_active

    # Deterministic per-particle jitter (small, to break the perfect-grid look)
    jitters = [
        (rng.uniform(-0.25, 0.25), rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2))
        for _ in range(20)
    ]

    def lerp3(a, b, t):
        return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t)

    def active_pos(target, i, n, frac, jitter):
        """Position of the i-th particle in an n-particle group, at conveyor fraction frac."""
        t = min(1.0, i / max(n, 1) + frac)
        p = lerp3(SOMA, target, t)
        return (round(p[0] + jitter[0]*0.2, 3),
                round(p[1] + jitter[1]*0.15, 3),
                round(p[2] + jitter[2]*0.2, 3))

    def rest_pos(jitter):
        return (round(SOMA[0] + jitter[0]*0.3, 3),
                round(SOMA[1] + jitter[1]*0.2 - 0.2, 3),
                round(SOMA[2] + jitter[2]*0.3, 3))

    def snapshot(step_frac):
        """All 20 particle positions at the given conveyor step fraction."""
        pos = []
        for i in range(n_d1):
            pos.append(active_pos(D1_LOCAL, i, n_d1, step_frac, jitters[i]))
        for i in range(n_d2):
            pos.append(active_pos(D2_LOCAL, i, n_d2, step_frac, jitters[n_d1+i]))
        for i in range(n_rest):
            pos.append(rest_pos(jitters[n_d1+n_d2+i]))
        return pos

    def fmt_positions(positions):
        """Format 20 positions into multi-line USDA array rows, 4 per row."""
        rows = []
        for chunk_start in range(0, 20, 4):
            chunk = positions[chunk_start:chunk_start+4]
            row = ", ".join(f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})" for p in chunk)
            rows.append(f"                        {row},")
        return rows

    # ── Build keyframe dict ────────────────────────────────────────────────────
    if dopamine < 0.02:
        # Parkinsonian: particles frozen near soma
        base = snapshot(0.0)
        kf = {0: base, 240: base}
    else:
        T    = max(30, int(120 * (1.0 - dopamine * 0.70)))   # loop period in frames
        step = 1.0 / max(n_d1, n_d2, 1)                       # path fraction per loop
        kf   = {}
        f    = 0
        while f <= 240:
            kf[f] = snapshot(0.0)               # loop start: particles at base positions
            f_end = f + T
            if f_end <= 240:
                kf[f_end] = snapshot(step)      # loop end: lead particle at MSN target
                if f_end + 1 <= 240:
                    kf[f_end + 1] = snapshot(0.0)  # hard reset (disguised by density)
            else:
                # Partial final loop: interpolate to frame 240
                partial = step * (240 - f) / T
                kf[240] = snapshot(partial)
                break
            f = f_end + 1

    # ── Render to USDA lines ───────────────────────────────────────────────────
    kf_lines = []
    for frame in sorted(kf.keys()):
        kf_lines.append(f"                    {frame}: [")
        kf_lines.extend(fmt_positions(kf[frame]))
        kf_lines.append(f"                    ],")

    return [
        "",
        "    # DopamineParticles flow: vesicles travel from SNc toward D1/D2 MSN targets.",
        "    # Density (n_active) and speed (T) both scale with the dopamine level.",
        "    # Overrides the static Y-only animation in animation.usda.",
        '    over "SNc"',
        "    {",
        '        over "DopamineParticles"',
        "        {",
        "            point3f[] positions.timeSamples = {",
        *kf_lines,
        "            }",
        "        }",
        "    }",
    ]


def _snc_combined_lines(dopamine: float) -> list:
    """
    Single over "SNc" block containing Soma displayColor, NigrostriatalAxon
    displayColor, and DopamineParticles positions — avoids duplicate-prim errors.

    The NigrostriatalAxon (ascending dopamine axon) pulses with the same timing
    as the soma, making the dopamine pathway visually traceable from SNc upward
    through the nigrostriatal tract to the striatum.
    """
    soma_lines = _snc_dopamine_lines(dopamine)
    particles_full = _dopamine_particles_lines(dopamine)
    particles_inner = particles_full[6:-1]   # strip outer over "SNc" wrapper

    # NigrostriatalAxon glow — same rhythm as soma but slightly brighter peak
    # so the ascending dopamine pathway is clearly visible
    pw = max(6, int(dopamine * 25))
    dim_r = round(max(0.2, 0.2 + dopamine * 0.55), 3)
    dim_g = round(max(0.3, 0.3 + dopamine * 0.65), 3)
    peak_r = round(min(1.0, dim_r + 0.42), 3)
    peak_g = round(min(1.0, dim_g + 0.32), 3)
    axon_kf = [
        (0,            dim_r,  dim_g,  0.02),
        (pw,           peak_r, peak_g, 0.10),
        (pw * 2,       dim_r,  dim_g,  0.02),
        (80,           dim_r,  dim_g,  0.02),
        (80 + pw,      peak_r, peak_g, 0.10),
        (80 + pw * 2,  dim_r,  dim_g,  0.02),
        (160,          dim_r,  dim_g,  0.02),
        (160 + pw,     peak_r, peak_g, 0.10),
        (160 + pw * 2, dim_r,  dim_g,  0.02),
        (240,          dim_r,  dim_g,  0.02),
    ]
    axon_full = _soma_color_lines(["SNc", "Axon", "NigrostriatalAxon"], axon_kf)
    axon_inner = axon_full[2:-1]   # strip 'over "SNc"' + '{' and closing '}'

    return [
        "",
        "    # SNc: Soma pulse + NigrostriatalAxon glow + DopamineParticles (one over block).",
        *soma_lines[:-1],    # open over "SNc", Soma content — NOT closing '    }'
        *axon_inner,         # NigrostriatalAxon glow at 8-space indent
        *particles_inner,    # DopamineParticles at 8-space indent
        soma_lines[-1],      # closing '    }'
    ]


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
        "    # Anatomy glow: dendrites → soma → axon light up per neuron in sequence.",
        "    # _OverTree merges all overrides for the same prim path — no duplicate prims.",
    ]

    # ── keyframes ──────────────────────────────────────────────────────────────
    kf_cx_dend = [  # cortex dendrites: pink glow (activation)
        (0, 0.88, 0.72, 0.76), (6, 1.0, 0.55, 0.65),
        (22, 1.0, 0.55, 0.65), (36, 0.88, 0.72, 0.76),
    ]
    kf_cx_axon = [  # cortex axon: orange (glutamate leaving)
        (17, 0.75, 0.70, 0.88), (22, 1.0, 0.58, 0.22),
        (55, 0.95, 0.52, 0.18), (67, 0.75, 0.70, 0.88),
    ]
    kf_d1_soma = [  # D1 MSN soma: blue flash
        (55, 0.2, 0.4, 1.0), (65, 0.5, 0.7, 1.0),
        (80, 0.3, 0.5, 1.0), (90, 0.2, 0.4, 1.0),
    ]
    kf_d1_dend = [  # D1 dendrites: dopamine yellow-green → glut orange → D1 blue
        (0,  0.48, 0.82, 0.78),
        (7,  0.58, 0.88, 0.42),   # dopamine background (D1R facilitation)
        (45, 0.58, 0.88, 0.42),
        (52, 0.48, 0.82, 0.78),
        (57, 0.92, 0.52, 0.22),   # orange: glutamate from cortex
        (64, 0.25, 0.52, 1.0),    # blue: D1 receptor activation
        (80, 0.28, 0.52, 1.0),
        (91, 0.48, 0.82, 0.78),
    ]
    kf_d1_axon = [  # D1 axon: blue GABA to GPi
        (77, 0.44, 0.60, 0.88), (82, 0.22, 0.48, 1.0),
        (118, 0.22, 0.48, 1.0), (128, 0.44, 0.60, 0.88),
    ]
    kf_d2_dend = [  # D2 dendrites: dopamine arrives then dims (D2R inhibition)
        (0,  0.48, 0.82, 0.78),
        (7,  0.58, 0.88, 0.42),
        (18, 0.32, 0.50, 0.35),   # dims: D2R inhibition suppresses NoGo
        (45, 0.32, 0.50, 0.35),
        (53, 0.48, 0.82, 0.78),
    ]
    kf_gpi_soma = [  # GPi soma: dims as D1 GABA suppresses it
        (110, 0.75, 0.6, 0.9), (125, 0.2, 0.1, 0.3),
        (145, 0.03, 0.01, 0.06), (180, 0.75, 0.6, 0.9),
    ]
    kf_gpi_dend = [  # GPi dendrites: blue flash (receiving D1 GABA), then dark
        (107, 0.78, 0.65, 0.92), (113, 0.32, 0.42, 1.0),
        (123, 0.06, 0.02, 0.10), (150, 0.78, 0.65, 0.92),
    ]
    kf_gpi_axon = [  # GPi axon: dims (suppressed, output stops)
        (127, 0.65, 0.55, 0.82), (134, 0.04, 0.02, 0.06),
        (178, 0.04, 0.02, 0.06), (190, 0.65, 0.55, 0.82),
    ]
    kf_th_soma = [  # Thalamus soma: gold flash (released!)
        (140, 0.75, 0.6, 0.9), (152, 1.0, 0.92, 0.6),
        (168, 0.9, 0.8, 0.5), (195, 0.75, 0.6, 0.9),
    ]
    kf_th_dend = [  # Thalamus dendrites: gold glow
        (137, 0.78, 0.65, 0.92), (143, 1.0, 0.92, 0.52),
        (175, 1.0, 0.92, 0.52), (200, 0.78, 0.65, 0.92),
    ]
    kf_th_axon = [  # Thalamus axon: gold (motor command to cortex)
        (167, 0.65, 0.55, 0.82), (173, 1.0, 0.90, 0.42),
        (205, 1.0, 0.90, 0.42), (215, 0.65, 0.55, 0.82),
    ]

    # ── Build override tree — every prim path registered exactly once ──────────
    t = _OverTree()

    # Cortex
    t.add(["Cortex", "PyramidalNeuron", "ApicalDendrite", "ApicalTrunk"], kf_cx_dend)
    for n in _PYRAMIDAL_BASAL_NAMES:
        t.add(["Cortex", "PyramidalNeuron", "BasalDendrites", n], kf_cx_dend)
    t.add(["Cortex", "PyramidalNeuron", "Axon", "MainAxon"], kf_cx_axon)

    # Striatum: D1 (soma + dendrites + axon) + D2 (dendrites: dopamine suppression)
    t.add(["Striatum", "D1_MSN", "Soma"], kf_d1_soma)
    for n in _MSN_DENDRITE_NAMES:
        t.add(["Striatum", "D1_MSN", "Dendrites", n], kf_d1_dend)
    t.add(["Striatum", "D1_MSN", "Axon", "MainAxon"], kf_d1_axon)
    for n in _MSN_DENDRITE_NAMES:
        t.add(["Striatum", "D2_MSN", "Dendrites", n], kf_d2_dend)

    # GPi: soma (dims) + dendrites (blue flash from D1 GABA) + axon (stops)
    t.add(["GPi", "GPi_Neuron", "Soma"], kf_gpi_soma)
    for n in _GENERIC_DENDRITE_NAMES:
        t.add(["GPi", "GPi_Neuron", "Dendrites", n], kf_gpi_dend)
    t.add(["GPi", "GPi_Neuron", "Axon", "MainAxon"], kf_gpi_axon)

    # Thalamus: soma (gold) + dendrites (gold) + axon (gold motor command)
    t.add(["Thalamus", "Thalamic_Neuron", "Soma"], kf_th_soma)
    for n in _GENERIC_DENDRITE_NAMES:
        t.add(["Thalamus", "Thalamic_Neuron", "Dendrites", n], kf_th_dend)
    t.add(["Thalamus", "Thalamic_Neuron", "Axon", "MainAxon"], kf_th_axon)

    # ── Connection curve glows — direct pathway ────────────────────────────────
    # Orange: glutamate axons pulse as the cortical signal travels.
    # Blue:   D1 GABAergic axon glows as it suppresses GPi.
    # Gold:   GPi→Thalamus flashes gold (disinhibition = gate opens).
    # Dopamine: SNc axons pulse with each dopamine release event.
    _CX_R = (0.9, 0.5, 0.2);   _CX_P = (1.0, 0.78, 0.38)
    _D1_R = (0.4, 0.3, 0.8);   _D1_P = (0.3, 0.55, 1.0)
    _GD_R = (0.4, 0.3, 0.8);   _GD_P = (1.0, 0.92, 0.42)
    _da   = values["dopamine"]
    _pw   = max(6, int(_da * 25))
    _DA_R = (0.55, 0.72, 0.12)
    _DA_P = (min(1.0, 0.6 + _da * 0.4), 1.0, 0.22)
    t.add(["Connections", "Cortex_to_D1"],    [(0, *_CX_R), (19, *_CX_R), (24, *_CX_P), (56, *_CX_P), (67, *_CX_R), (240, *_CX_R)])
    t.add(["Connections", "Cortex_to_STN"],   [(0, *_CX_R), (22, *_CX_R), (27, *_CX_P), (62, *_CX_P), (73, *_CX_R), (240, *_CX_R)])
    t.add(["Connections", "D1_to_GPi"],       [(0, *_D1_R), (79, *_D1_R), (84, *_D1_P), (119, *_D1_P), (131, *_D1_R), (240, *_D1_R)])
    t.add(["Connections", "GPi_to_Thalamus"], [(0, *_GD_R), (139, *_GD_R), (144, *_GD_P), (179, *_GD_P), (192, *_GD_R), (240, *_GD_R)])
    _kf_snc = [(0, *_DA_R), (_pw, *_DA_P), (_pw * 2, *_DA_R),
               (80, *_DA_R), (80 + _pw, *_DA_P), (80 + _pw * 2, *_DA_R),
               (160, *_DA_R), (160 + _pw, *_DA_P), (160 + _pw * 2, *_DA_R), (240, *_DA_R)]
    t.add(["Connections", "SNc_to_D1"], _kf_snc)
    t.add(["Connections", "SNc_to_D2"], _kf_snc)

    lines += t.to_lines()

    # ── SNc: soma + axon + particles (merged by _snc_combined_lines) ──────────
    lines += _snc_combined_lines(values["dopamine"])

    return lines


def _indirect_pathway_lines(values: dict) -> list:
    """
    USDA body lines for an Indirect (NoGo) pathway win.

    Visual story (full indirect chain, frames 20–220):
      [20-60]   CortexToStriatum orb → D2 MSN (4,10,0) — cortex activates NoGo striatum.
      [55-90]   D2 MSN flashes amber (firing).
      [80-110]  D2ToGPe orb → GPe (5,2,0) — D2 GABA suppresses GPe.
      [85-155]  GPe darkens (suppressed, releases STN from inhibition).
      [110-130] GPeToSTN orb → STN (8,-4,2) — lavender = GPe's grip draining away.
      [100-165] STN flashes teal (disinhibited, drives GPi).
      [120-160] STNToGPi orb → GPi (-5,2,0) — glutamate drives GPi up.
      [120-200] GPi stays bright purple (hyperactive, movement blocked).
      [162-200] GPiToThalamus_suppress orb → Thalamus (0,-10,0) — purple = inhibition.
      [198-215] Thalamus dims briefly (suppression hit), stays dark — no action output.
      Direct-pathway pulse orbs zeroed so they don't play.
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
        "        # GPe → STN disinhibition: as GPe goes dark (D2 MSN suppressed it),",
        "        # its GABAergic hold on STN lifts. Lavender orb = GPe's inhibition draining away.",
        "        # Arrives at STN just before STNToGPi fires — STN wakes up and immediately",
        "        # starts driving GPi.",
        "        def Sphere \"GPeToSTN\"",
        "        {",
        "            double radius = 0.2",
        "            color3f[] primvars:displayColor = [(0.75, 0.6, 0.9)]",
        "            double3 xformOp:translate.timeSamples = {",
        "                110: (5.0, 2.0, 0.0),",
        "                120: (6.5, -1.0, 1.0),",
        "                130: (8.0, -4.0, 2.0)",
        "            }",
        "            double3 xformOp:scale.timeSamples = {",
        "                109: (0.0, 0.0, 0.0),",
        "                110: (1.0, 1.0, 1.0),",
        "                129: (1.0, 1.0, 1.0),",
        "                130: (0.0, 0.0, 0.0)",
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
        "",
        "        # GPi → Thalamus suppression: STN has driven GPi to full activity.",
        "        # GPi fires its GABAergic suppression signal at Thalamus — movement blocked.",
        "        # Bright purple = GPi hyperactive. Same path as the direct-pathway release",
        "        # orb (GPiToThalamus in animation.usda) but in GPi's color, not gold.",
        "        def Sphere \"GPiToThalamus_suppress\"",
        "        {",
        "            double radius = 0.2",
        "            color3f[] primvars:displayColor = [(0.8, 0.4, 1.0)]",
        "            double3 xformOp:translate.timeSamples = {",
        "                162: (-5.0, 2.0, 0.0),",
        "                181: (-2.5, -4.0, 0.0),",
        "                200: (0.0, -10.0, 0.0)",
        "            }",
        "            double3 xformOp:scale.timeSamples = {",
        "                161: (0.0, 0.0, 0.0),",
        "                162: (1.0, 1.0, 1.0),",
        "                199: (1.0, 1.0, 1.0),",
        "                200: (0.0, 0.0, 0.0)",
        "            }",
        "            uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]",
        "        }",
        "    }",
    ]

    # ── keyframes ──────────────────────────────────────────────────────────────
    kf_cx_dend = [
        (0, 0.88, 0.72, 0.76), (6, 1.0, 0.55, 0.65),
        (22, 1.0, 0.55, 0.65), (36, 0.88, 0.72, 0.76),
    ]
    kf_cx_axon = [
        (17, 0.75, 0.70, 0.88), (22, 1.0, 0.58, 0.22),
        (55, 0.95, 0.52, 0.18), (67, 0.75, 0.70, 0.88),
    ]
    kf_d2_soma = [(55, 1.0, 0.6, 0.1), (65, 1.0, 0.85, 0.2), (80, 1.0, 0.7, 0.15), (90, 1.0, 0.6, 0.1)]
    kf_d2_dend = [
        (0,  0.48, 0.82, 0.78), (7,  0.58, 0.88, 0.42), (52, 0.48, 0.82, 0.78),
        (57, 0.92, 0.52, 0.22), (64, 1.0,  0.72, 0.18), (80, 1.0, 0.72, 0.18),
        (91, 0.48, 0.82, 0.78),
    ]
    kf_d2_axon = [(77, 0.44, 0.60, 0.88), (82, 1.0, 0.65, 0.15), (108, 1.0, 0.65, 0.15), (118, 0.44, 0.60, 0.88)]
    kf_d1_dend = [(0, 0.48, 0.82, 0.78), (7, 0.58, 0.88, 0.42), (45, 0.58, 0.88, 0.42), (53, 0.48, 0.82, 0.78)]
    kf_gpe_soma = [(85, 0.75, 0.6, 0.9), (100, 0.2, 0.1, 0.3), (118, 0.03, 0.01, 0.06), (155, 0.75, 0.6, 0.9)]
    kf_gpe_dend = [(82, 0.78, 0.65, 0.92), (88, 1.0, 0.70, 0.18), (100, 0.06, 0.02, 0.08), (155, 0.78, 0.65, 0.92)]
    kf_gpe_axon = [(98, 0.65, 0.55, 0.82), (105, 0.04, 0.02, 0.06), (155, 0.04, 0.02, 0.06), (165, 0.65, 0.55, 0.82)]
    kf_stn_soma = [(100, 0.75, 0.6, 0.9), (118, 0.0, 0.85, 0.85), (135, 0.0, 0.65, 0.65), (165, 0.75, 0.6, 0.9)]
    kf_stn_dend = [(98, 0.78, 0.65, 0.92), (108, 0.80, 0.62, 0.95), (120, 0.0, 0.88, 0.88), (145, 0.0, 0.88, 0.88), (167, 0.78, 0.65, 0.92)]
    kf_stn_axon = [(118, 0.65, 0.55, 0.82), (124, 0.0, 0.88, 0.88), (158, 0.0, 0.88, 0.88), (168, 0.65, 0.55, 0.82)]
    kf_gpi_soma = [(120, 0.6, 0.3, 0.85), (160, 0.6, 0.3, 0.85), (200, 0.75, 0.6, 0.9)]
    kf_gpi_dend = [(128, 0.78, 0.65, 0.92), (135, 0.0, 0.88, 0.88), (148, 0.62, 0.32, 0.88), (200, 0.62, 0.32, 0.88)]
    kf_gpi_axon = [(128, 0.65, 0.55, 0.82), (135, 0.78, 0.38, 1.0), (200, 0.78, 0.38, 1.0), (215, 0.65, 0.55, 0.82)]
    kf_th_soma = [(0, 0.3, 0.2, 0.4), (198, 0.3, 0.2, 0.4), (205, 0.12, 0.06, 0.18), (215, 0.3, 0.2, 0.4), (240, 0.3, 0.2, 0.4)]
    kf_th_dend = [(0, 0.35, 0.25, 0.45), (197, 0.35, 0.25, 0.45), (203, 0.16, 0.08, 0.22), (215, 0.35, 0.25, 0.45), (240, 0.35, 0.25, 0.45)]

    # ── Build override tree — every prim path registered exactly once ──────────
    t = _OverTree()

    # Cortex: dendrites + axon glow
    t.add(["Cortex", "PyramidalNeuron", "ApicalDendrite", "ApicalTrunk"], kf_cx_dend)
    for n in _PYRAMIDAL_BASAL_NAMES:
        t.add(["Cortex", "PyramidalNeuron", "BasalDendrites", n], kf_cx_dend)
    t.add(["Cortex", "PyramidalNeuron", "Axon", "MainAxon"], kf_cx_axon)

    # Striatum: D2 (soma + dendrites + axon) + D1 (dendrites: dopamine visible)
    t.add(["Striatum", "D2_MSN", "Soma"], kf_d2_soma)
    for n in _MSN_DENDRITE_NAMES:
        t.add(["Striatum", "D2_MSN", "Dendrites", n], kf_d2_dend)
    t.add(["Striatum", "D2_MSN", "Axon", "MainAxon"], kf_d2_axon)
    for n in _MSN_DENDRITE_NAMES:
        t.add(["Striatum", "D1_MSN", "Dendrites", n], kf_d1_dend)

    # GPe: soma (dims) + dendrites (amber flash) + axon (dims — releases STN)
    t.add(["GPe", "GPe_Neuron", "Soma"], kf_gpe_soma)
    for n in _GENERIC_DENDRITE_NAMES:
        t.add(["GPe", "GPe_Neuron", "Dendrites", n], kf_gpe_dend)
    t.add(["GPe", "GPe_Neuron", "Axon", "MainAxon"], kf_gpe_axon)

    # STN: soma (teal) + dendrites (lavender→teal) + axon (teal glutamate to GPi)
    t.add(["STN", "STN_Neuron", "Soma"], kf_stn_soma)
    for n in _GENERIC_DENDRITE_NAMES:
        t.add(["STN", "STN_Neuron", "Dendrites", n], kf_stn_dend)
    t.add(["STN", "STN_Neuron", "Axon", "MainAxon"], kf_stn_axon)

    # GPi: soma (purple, stays) + dendrites (teal→purple from STN) + axon (purple, active)
    t.add(["GPi", "GPi_Neuron", "Soma"], kf_gpi_soma)
    for n in _GENERIC_DENDRITE_NAMES:
        t.add(["GPi", "GPi_Neuron", "Dendrites", n], kf_gpi_dend)
    t.add(["GPi", "GPi_Neuron", "Axon", "MainAxon"], kf_gpi_axon)

    # Thalamus: soma (suppressed) + dendrites (brief dim on GPi hit)
    t.add(["Thalamus", "Thalamic_Neuron", "Soma"], kf_th_soma)
    for n in _GENERIC_DENDRITE_NAMES:
        t.add(["Thalamus", "Thalamic_Neuron", "Dendrites", n], kf_th_dend)

    # ── Connection curve glows — indirect pathway ──────────────────────────────
    # Orange:   cortex → D2 MSN + cortex → STN (hyperdirect) light up.
    # Amber:    D2 MSN → GPe (GABA suppressing GPe).
    # Lavender: GPe → STN (disinhibition: GPe's grip lifts from STN).
    # Teal:     STN → GPi (glutamate drives GPi up — NoGo amplified).
    # Purple:   GPi → Thalamus flashes bright purple (movement blocked!).
    _CX_R = (0.9, 0.5, 0.2);   _CX_P = (1.0, 0.78, 0.38)
    _D2_R = (0.4, 0.3, 0.8);   _D2_P = (1.0, 0.72, 0.2)
    _LV_R = (0.4, 0.3, 0.8);   _LV_P = (0.8, 0.65, 1.0)
    _TL_R = (0.4, 0.3, 0.8);   _TL_P = (0.0, 0.9, 0.9)
    _GI_R = (0.4, 0.3, 0.8);   _GI_P = (0.75, 0.38, 1.0)
    _da   = values["dopamine"]
    _pw   = max(6, int(_da * 25))
    _DA_R = (0.55, 0.72, 0.12)
    _DA_P = (min(1.0, 0.6 + _da * 0.4), 1.0, 0.22)
    t.add(["Connections", "Cortex_to_D2"],    [(0, *_CX_R), (19, *_CX_R), (24, *_CX_P), (56, *_CX_P), (67, *_CX_R), (240, *_CX_R)])
    t.add(["Connections", "Cortex_to_STN"],   [(0, *_CX_R), (22, *_CX_R), (27, *_CX_P), (62, *_CX_P), (73, *_CX_R), (240, *_CX_R)])
    t.add(["Connections", "D2_to_GPe"],       [(0, *_D2_R), (79, *_D2_R), (84, *_D2_P), (109, *_D2_P), (121, *_D2_R), (240, *_D2_R)])
    t.add(["Connections", "GPe_to_STN"],      [(0, *_LV_R), (109, *_LV_R), (113, *_LV_P), (129, *_LV_P), (141, *_LV_R), (240, *_LV_R)])
    t.add(["Connections", "STN_to_GPi"],      [(0, *_TL_R), (119, *_TL_R), (124, *_TL_P), (159, *_TL_P), (171, *_TL_R), (240, *_TL_R)])
    t.add(["Connections", "GPi_to_Thalamus"], [(0, *_GI_R), (161, *_GI_R), (165, *_GI_P), (199, *_GI_P), (211, *_GI_R), (240, *_GI_R)])
    _kf_snc = [(0, *_DA_R), (_pw, *_DA_P), (_pw * 2, *_DA_R),
               (80, *_DA_R), (80 + _pw, *_DA_P), (80 + _pw * 2, *_DA_R),
               (160, *_DA_R), (160 + _pw, *_DA_P), (160 + _pw * 2, *_DA_R), (240, *_DA_R)]
    t.add(["Connections", "SNc_to_D1"], _kf_snc)
    t.add(["Connections", "SNc_to_D2"], _kf_snc)

    lines += t.to_lines()

    # ── SNc: soma + axon + particles (merged by _snc_combined_lines) ──────────
    lines += _snc_combined_lines(values["dopamine"])

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
