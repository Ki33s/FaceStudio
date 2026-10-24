"""Smooth measured landmarks without averaging different head poses."""
import numpy as np


def pose(points):
    points=np.asarray(points,dtype=float)
    axis=points[1]-points[0]
    width=max(float(np.linalg.norm(axis)),1.)
    axis/=width
    eyes=(points[0]+points[1])*.5
    mouth=(points[3]+points[4])*.5
    height=max(float(np.linalg.norm(mouth-eyes)),1.)
    return float((points[2]-eyes)@axis/height),width/height,height


def smooth_measured(anchor,samples,weights):
    anchor=np.asarray(anchor,dtype=float)
    yaw,spread,scale=pose(anchor)
    accepted=[anchor];scores=[max(float(np.max(weights)),.15)]
    for points,weight in zip(samples,weights):
        points=np.asarray(points,dtype=float)
        other_yaw,other_spread,_=pose(points)
        # A motion transform cannot undo an actual turn of the head. In a
        # profile, inter-eye distance is compressed and is not a useful scale.
        if abs(other_yaw-yaw)>.20 or abs(other_spread-spread)>.35:continue
        if yaw*other_yaw<0 and min(abs(yaw),abs(other_yaw))>.12:continue
        residual=float(np.linalg.norm(points-anchor,axis=1).max())
        if residual>.15*scale:continue
        accepted.append(points)
        scores.append(float(weight)*np.exp(-.5*(residual/(.04*scale))**2))
    result=np.average(accepted,axis=0,weights=scores)
    delta=result-anchor
    largest=float(np.linalg.norm(delta,axis=1).max())
    if largest>.04*scale:result=anchor+delta*(.04*scale/largest)
    return result.astype(np.float32)
