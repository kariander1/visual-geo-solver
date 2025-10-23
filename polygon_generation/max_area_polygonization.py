#!/usr/bin/env python3
"""
max_area_polygonization.py

Exact backtracking solver for Maximum-Area simple polygonization (n ≲ 10).
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import math, sys, time

Point = Tuple[float,float]
EPS = 1e-12

def orientation(ax, ay, bx, by, cx, cy):
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

def on_segment(ax, ay, bx, by, cx, cy):
    return (min(ax,bx)-EPS <= cx <= max(ax,bx)+EPS and
            min(ay,by)-EPS <= cy <= max(ay,by)+EPS)

def segments_properly_intersect(a:Point,b:Point,c:Point,d:Point)->bool:
    ax,ay=a; bx,by=b; cx,cy=c; dx,dy=d
    o1=orientation(ax,ay,bx,by,cx,cy); o2=orientation(ax,ay,bx,by,dx,dy)
    o3=orientation(cx,cy,dx,dy,ax,ay); o4=orientation(cx,cy,dx,dy,bx,by)
    if (o1*o2<-EPS) and (o3*o4<-EPS): return True
    if abs(o1)<=EPS and on_segment(ax,ay,bx,by,cx,cy):
        if (abs(cx-ax)>EPS or abs(cy-ay)>EPS) and (abs(cx-bx)>EPS or abs(cy-by)>EPS): return True
    if abs(o2)<=EPS and on_segment(ax,ay,bx,by,dx,dy):
        if (abs(dx-ax)>EPS or abs(dy-ay)>EPS) and (abs(dx-bx)>EPS or abs(dy-by)>EPS): return True
    if abs(o3)<=EPS and on_segment(cx,cy,dx,dy,ax,ay):
        if (abs(ax-cx)>EPS or abs(ay-cy)>EPS) and (abs(ax-dx)>EPS or abs(ay-dy)>EPS): return True
    if abs(o4)<=EPS and on_segment(cx,cy,dx,dy,bx,by):
        if (abs(bx-cx)>EPS or abs(by-cy)>EPS) and (abs(bx-dx)>EPS or abs(by-dy)>EPS): return True
    return False

def polygon_area(points:List[Point], order:List[int])->float:
    area2=0.0; n=len(order)
    for i in range(n):
        x1,y1=points[order[i]]; x2,y2=points[order[(i+1)%n]]
        area2 += x1*y2 - x2*y1
    return abs(area2)*0.5

def area_sign(points:List[Point], order:List[int])->float:
    area2=0.0; n=len(order)
    for i in range(n):
        x1,y1=points[order[i]]; x2,y2=points[order[(i+1)%n]]
        area2 += x1*y2 - x2*y1
    return area2*0.5

def _choose_anchor(points:List[Point])->int:
    best=0; bx,by=points[0][0],points[0][1]
    for i,(x,y) in enumerate(points):
        if (y<by-EPS) or (abs(y-by)<=EPS and x<bx-EPS):
            best=i; bx,by=x,y
    return best

def _precompute_cross(points:List[Point]):
    n=len(points)
    cross=[[[[False]*n for _ in range(n)] for __ in range(n)] for ___ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            a=points[i]; b=points[j]
            for k in range(n):
                for l in range(n):
                    if k==l or i in (k,l) or j in (k,l): continue
                    if segments_properly_intersect(a,b,points[k],points[l]):
                        cross[i][j][k][l]=True
    return cross

def max_area_polygon(points:List[Point], time_limit:Optional[float]=None, precompute_cross:bool=True):
    n=len(points); assert n>=3
    start=time.time()
    anchor=_choose_anchor(points)
    cross=_precompute_cross(points) if precompute_cross else None

    used=[False]*n; used[anchor]=True
    path=[anchor]
    best_order=[]; best_area=-1.0

    cx=sum(x for x,_ in points)/n; cy=sum(y for _,y in points)/n
    angles=[math.atan2(y-cy,x-cx) for (x,y) in points]
    angle_order=sorted(range(n), key=lambda i: angles[i])

    def last_edge_intersects(v:int)->bool:
        if len(path)<2: return False
        u=path[-1]
        if cross is not None:
            for i in range(len(path)-2):
                p=path[i]; q=path[i+1]
                if cross[u][v][p][q]: return True
        else:
            a=points[u]; b=points[v]
            for i in range(len(path)-2):
                p=path[i]; q=path[i+1]
                if segments_properly_intersect(a,b,points[p],points[q]): return True
        return False

    def closing_edge_intersects()->bool:
        u=path[-1]; v=path[0]
        if cross is not None:
            for i in range(len(path)-1):
                p=path[i]; q=path[i+1]
                if p in (u,v) or q in (u,v): continue
                if cross[u][v][p][q]: return True
        else:
            a=points[u]; b=points[v]
            for i in range(len(path)-1):
                p=path[i]; q=path[i+1]
                if p in (u,v) or q in (u,v): continue
                if segments_properly_intersect(a,b,points[p],points[q]): return True
        return False

    def dfs():
        nonlocal best_order,best_area,start
        if time_limit is not None and (time.time()-start)>time_limit: return
        if len(path)==n:
            if closing_edge_intersects(): return
            s=area_sign(points, path)
            if s<=EPS: return
            area=abs(s)
            if area>best_area+1e-15:
                best_area=area; best_order=path.copy()
            return
        for v in angle_order:
            if used[v]: continue
            if len(path)==n-1 and v==anchor: continue
            if last_edge_intersects(v): continue
            used[v]=True; path.append(v)
            dfs()
            path.pop(); used[v]=False

    dfs()
    return best_order, best_area

def _read_points_stdin():
    pts=[]
    for line in sys.stdin:
        s=line.strip()
        if not s or s.startswith("#"): continue
        xs=s.replace(",", " ").split()
        if len(xs)<2: continue
        pts.append((float(xs[0]), float(xs[1])))
    return pts

def main():
    pts=_read_points_stdin()
    order, area = max_area_polygon(pts)
    print("# Best order (0-based indices):")
    print(" ".join(map(str, order)))
    print("# Area:")
    print(f"{area:.12f}")
    print("# Ordered coordinates:")
    for i in order:
        x,y=pts[i]
        print(f"{x} {y}")

if __name__=="__main__":
    main()
