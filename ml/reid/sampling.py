import random
from collections import defaultdict

def pk_batches(rows,steps,seed,identities=8,crops=4):
 rng=random.Random(seed);by=defaultdict(list);matches=defaultdict(list);teams=defaultdict(list)
 for i,r in enumerate(rows):by[r['player_identity']].append(i)
 for key,indices in by.items():matches[rows[indices[0]]['match_id']].append(key);teams[rows[indices[0]]['team_identity']].append(key)
 if len(by)<identities:raise ValueError('Insufficient independent identities for PK sampling')
 for _ in range(steps):
  match=rng.choice(sorted(matches));anchor=rng.choice(matches[match]);team=rows[by[anchor][0]]['team_identity'];local=[k for k in matches[match] if rows[by[k][0]]['team_identity']==team];rng.shuffle(local);other_same=[k for k in teams[team] if k not in local];rng.shuffle(other_same);same=local+other_same;other=[k for k in by if k not in same];chosen=same[:identities-1] if other else same[:identities]
  if other:chosen+=rng.sample(other,identities-len(chosen))
  if len(chosen)<identities:raise ValueError('Insufficient PK identities')
  batch=[]
  for key in chosen:
   ids=sorted(by[key],key=lambda i:rows[i]['timestamp']);bins=[ids[j*len(ids)//crops:(j+1)*len(ids)//crops] for j in range(crops)];batch.extend(rng.choice(b) for b in bins)
  yield batch
