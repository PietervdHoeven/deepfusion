from pathlib import Path
from collections import defaultdict

def dup_sessions(root="data/deepfusion/latents"):
    seen = defaultdict(set)
    for stage in ["train","val","test"]:
        p = Path(root, stage)
        for f in p.rglob("*_latent-maps.npy"):
            pid, sid, *_ = f.stem.split("_")
            seen[(pid, sid)].add(stage)
    return {k:v for k,v in seen.items() if len(v) > 1}

offenders = dup_sessions()
print("duplicate sessions across stages:", len(offenders))
for k,v in list(offenders.items())[:10]:
    print(k, "→", sorted(v))