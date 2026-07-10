#!/usr/bin/env python3
"""Copy positions.timeSamples from layers/dopamine_block.usda into basal_ganglia_hq.usda root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
hq = ROOT / "basal_ganglia_hq.usda"
dopa = ROOT / "layers/dopamine_block.usda"

text = hq.read_text()
dopa_text = dopa.read_text()
start = dopa_text.index("point3f[] positions.timeSamples")
end = dopa_text.index("            }", start) + len("            }")
ts_block = dopa_text[start:end]

import re
pattern = r'(        over "DopamineParticles"\s*\{)(.*?)(        \})'
match = re.search(pattern, text, re.DOTALL)
if not match:
    raise SystemExit("DopamineParticles block not found in basal_ganglia_hq.usda")

replacement = f'{match.group(1)}\n            active = true\n            {ts_block}\n{match.group(3)}'
text = text[: match.start()] + replacement + text[match.end() :]
hq.write_text(text)
print(f"Updated {hq}")
