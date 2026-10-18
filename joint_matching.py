"""One-to-one maximum-weight matching, with an explicit unmatched choice.

Accept an edge only if forbidding it makes the *whole* solution worse by the
requested margin. Local greedy row/column checks lose clear joint solutions.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(proposals, margin=.08, blocked_rows=(), blocked_columns=()):
    """Return (score, row, column) for confident pairs; never force a match."""
    blocked_rows=set(blocked_rows);blocked_columns=set(blocked_columns)
    edges={}
    for score,row,column in proposals:
        score=float(score)
        if row in blocked_rows or column in blocked_columns or not np.isfinite(score) or score<=0:continue
        edges[row,column]=max(score,edges.get((row,column),0))
    if not edges:return []
    # Dict insertion order supports both integer and string identifiers.
    rows=list(dict.fromkeys(row for row,_ in edges));columns=list(dict.fromkeys(col for _,col in edges))
    ri={v:i for i,v in enumerate(rows)};ci={v:i for i,v in enumerate(columns)}
    weights=np.full((len(rows),len(columns)+len(rows)),-1e6,np.float64)
    weights[:,len(columns):]=0  # Independent unmatched slots for every row.
    for (row,col),score in edges.items():weights[ri[row],ci[col]]=score
    a,b=linear_sum_assignment(-weights);best=float(weights[a,b].sum());result=[]
    for i,j in zip(a,b):
        if j>=len(columns) or weights[i,j]<=0:continue
        score=weights[i,j];weights[i,j]=-1e6
        x,y=linear_sum_assignment(-weights);alternative=float(weights[x,y].sum())
        weights[i,j]=score
        if best-alternative+1e-9>=margin:result.append((float(score),rows[i],columns[j]))
    return result
