Neuron Role Summary

  ┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────┬─────────────────────┬───────────────────────────┐
  │        Neuron        │                                       Role                                       │  Neurotransmitter   │        In Circuit         │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ Cortex               │ The "I want to move" INPUT signal. Sends excitation to both D1 and D2            │ Glutamate           │ Top of circuit, drives    │
  │ (PyramidalNeuron)    │ simultaneously                                                                   │ (excitatory)        │ everything                │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ D1 MSN               │ The GO decision-maker. When it fires, it SILENCES GPi → gate opens → movement    │ GABA (inhibitory)   │ Direct pathway left       │
  │                      │                                                                                  │                     │ branch                    │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ D2 MSN               │ The NoGo decision-maker. When it fires, it silences GPe → releases STN →         │ GABA (inhibitory)   │ Indirect pathway right    │
  │                      │ amplifies GPi → gate stays closed                                                │                     │ branch                    │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ GPe                  │ Indirect pathway relay. Normally suppresses STN; when D2 shuts it down, STN      │ GABA (inhibitory)   │ Right side, indirect only │
  │                      │ wakes up                                                                         │                     │                           │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ STN                  │ The ONLY excitatory nucleus. Gets input from both GPe (indirect) and cortex      │ Glutamate           │ Right side, crosses to    │
  │                      │ (hyperdirect). Always pushes GPi UP = movement suppression                       │ (excitatory)        │ GPi                       │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ GPi                  │ The final GATE. Fires at ~70Hz tonic, keeping thalamus locked. Go pathway        │ GABA (inhibitory)   │ Left side, convergence    │
  │                      │ silences it; NoGo pathway excites it. DBS target                                 │                     │ point                     │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ Thalamus             │ The motor command OUTPUT. Kept silent by GPi. When GPi is suppressed, thalamus   │ Glutamate           │ Bottom, motor output      │
  │                      │ fires → sends action command to cortex                                           │ (excitatory)        │                           │
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼─────────────────────┼───────────────────────────┤
  │ SNc                  │ Dopamine MODULATOR (not in the loop). Makes D1 more sensitive (promotes Go),     │ Dopamine            │ Background/depth          │
  │                      │ makes D2 less sensitive (suppresses NoGo). Loss = Parkinson's                    │                     │                           │
  └──────────────────────┴──────────────────────────────────────────────────────────────────────────────────┴─────────────────────┴───────────────────────────┘

  ---
  What the Animation Should Be Doing

  Default run (--strength 0.75 --dopamine 0.5) → Direct/Go pathway wins:

  Frame 0-30:    Cortex brightens (stimulus arrives)
  Frame 20-60:   Orange pulse orb: Cortex → D1 MSN
  Frame 55-90:   D1 MSN flashes BLUE (firing, sending GABA to GPi)
  Frame 80-120:  Blue orb: D1 MSN → GPi
  Frame 110-150: GPi DIMS and goes dark (gate opens!)
  Frame 140-180: Gold orb: GPi → Thalamus (the "release" signal)
  Frame 135-200: Thalamus glows GOLD (action released, movement!)
  Throughout:    SNc pulses yellow-green + dopamine particles rise

  With low dopamine (--dopamine 0.1) → Indirect/NoGo wins:

  Frame 20-60:   Orange orb redirected to D2 MSN instead
  Frame 55-90:   D2 MSN flashes AMBER
  Frame 80-110:  Amber orb: D2 MSN → GPe
  Frame 85-155:  GPe dims (STN released)
  Frame 110-130: Lavender orb: GPe → STN (disinhibition)
  Frame 100-165: STN flashes TEAL
  Frame 120-160: Teal orb: STN → GPi (excites it)
  Frame 120-200: GPi stays BRIGHT PURPLE (movement blocked!)
  Frame 162-200: Purple orb: GPi → Thalamus (suppression confirmed)
  Thalamus stays DARK — no action output

  ---
  Your Two Confusion Questions

  1. STN_to_GPi — "I thought the two routes were separate"

  They ARE separate until they converge at GPi — that's intentional and correct. GPi is the battlefield where both pathways compete:

  DIRECT:    Cortex → D1_MSN ──(GABA)──→ GPi  (pushes GPi DOWN → gate opens)
  INDIRECT:  Cortex → D2_MSN → GPe → STN ──(Glutamate)──→ GPi  (pushes GPi UP → gate closed)

  The STN_to_GPi curve going from the right side (STN at x=+8) to the left side (GPi at x=-5) IS the indirect pathway's final excitatory push on GPi. The
  information doesn't mix — D1's GABA inhibits GPi, STN's glutamate excites GPi. Two opposite inputs fighting for control of the same gate. That long crossing line
   is actually one of the most important features of the circuit to show visually — it illustrates that STN "reaches across" to strengthen the NoGo signal.

  2. Cortex_to_STN — What is this?

  This is the Hyperdirect Pathway — the third, fastest route (~6ms, vs ~30ms direct, ~40ms indirect). It bypasses the striatum entirely:

  Cortex ──(Glutamate)──→ STN ──(Glutamate)──→ GPi → suppresses Thalamus

  Purpose: a rapid BRAKING signal. Before the Go/NoGo competition in the striatum even resolves, the cortex sends a fast "hold on, pause everything" signal through
   STN. It's how you suppress competing actions — your cortex simultaneously says "do action A" through striatum AND "briefly stop all actions first" through the
  hyperdirect path, giving the circuit time to properly select the winner. In Parkinson's, this balance is disrupted.

