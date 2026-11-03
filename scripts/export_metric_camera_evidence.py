"""Export complete source-pixel landmark evidence for the original M7 long gap."""
import json
from pathlib import Path
import numpy as np
from scripts.diagnose_camera_m8 import groups

def run():
 root=Path('outputs/world_model/m8_verification/test_match'); rows=[json.loads(x) for x in (root/'release/timeline.jsonl').open()]
 cache=np.load('outputs/world_model/test_match_m2_pose/semantic_masks.npz');masks=cache['mask'];conf=cache['confidence']; previous=None
 with (root/'release/center_sequence_evidence.jsonl').open('w') as out:
  for row in rows:
   i=row['frame'];h=row['metric']['current_transform'];prev=previous
   if 151<=i<=455:
    g=groups(masks[i],conf[i]);row['landmark_pixels_source']={k:(v*[1080,1080]).tolist() for k,v in g.items()};row['pixel_units']='Original 1080x1080 source pixels, segmentation cell centres';row['previous_contiguous_accepted_transform']=prev;motion=row['field_motion'];row['predicted_transform_from_previous_and_measured_motion']=(np.asarray(prev)@np.linalg.inv(np.asarray(motion['homography']))).tolist() if prev is not None and motion.get('accepted') and 'homography' in motion else None;out.write(json.dumps(row)+'\n')
   previous=h if row['accepted'] else None
 cache.close()
if __name__=='__main__':run()
