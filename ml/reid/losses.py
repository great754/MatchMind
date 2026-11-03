import torch

def hard_triplet(embedding,labels,teams,margin=.3):
 d=1-embedding@embedding.T;equal=labels[:,None]==labels[None,:];eye=torch.eye(len(labels),device=labels.device,dtype=torch.bool);positive=equal&~eye;negative=~equal;same_team=teams[:,None]==teams[None,:];hard_negative=negative&same_team;has_hard=hard_negative.any(1);mask=torch.where(has_hard[:,None],hard_negative,negative);pos=d.masked_fill(~positive,-10).max(1).values;neg=d.masked_fill(~mask,10).min(1).values;valid=positive.any(1)&negative.any(1)
 return torch.relu(pos[valid]-neg[valid]+margin).mean() if valid.any() else embedding.sum()*0
