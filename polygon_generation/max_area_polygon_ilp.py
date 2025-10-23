#!/usr/bin/env python3
"""
max_area_polygon_ilp.py

Exact IP model for Maximum-Area Polygonization of a 2D point set.
Requires PuLP: `pip install pulp` (CBC by default; Gurobi/CPLEX optional).

CLI:
    python max_area_polygon_ilp.py < points.txt

Library:
    from max_area_polygon_ilp import solve_max_area_polygon_ilp
    order, area = solve_max_area_polygon_ilp(points, time_limit=10, solver_name="CBC")
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import sys

try:
    import pulp
except ImportError:
    print("PuLP is required. Install with: pip install pulp", file=sys.stderr)
    raise

Point = Tuple[float, float]
EPS = 1e-12

def orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def on_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    return (min(ax, bx) - EPS <= cx <= max(ax, bx) + EPS and
            min(ay, by) - EPS <= cy <= max(ay, by) + EPS)

def segments_properly_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
    o1 = orientation(ax, ay, bx, by, cx, cy)
    o2 = orientation(ax, ay, bx, by, dx, dy)
    o3 = orientation(cx, cy, dx, dy, ax, ay)
    o4 = orientation(cx, cy, dx, dy, bx, by)
    if (o1 * o2 < -EPS) and (o3 * o4 < -EPS):
        return True
    # Collinear overlap beyond endpoints counts as invalid (crossing)
    if abs(o1) <= EPS and on_segment(ax, ay, bx, by, cx, cy):
        if (abs(cx-ax) > EPS or abs(cy-ay) > EPS) and (abs(cx-bx) > EPS or abs(cy-by) > EPS):
            return True
    if abs(o2) <= EPS and on_segment(ax, ay, bx, by, dx, dy):
        if (abs(dx-ax) > EPS or abs(dy-ay) > EPS) and (abs(dx-bx) > EPS or abs(dy-by) > EPS):
            return True
    if abs(o3) <= EPS and on_segment(cx, cy, dx, dy, ax, ay):
        if (abs(ax-cx) > EPS or abs(ay-cy) > EPS) and (abs(ax-dx) > EPS or abs(ay-dy) > EPS):
            return True
    if abs(o4) <= EPS and on_segment(cx, cy, dx, dy, bx, by):
        if (abs(bx-cx) > EPS or abs(by-cy) > EPS) and (abs(bx-dx) > EPS or abs(by-dy) > EPS):
            return True
    return False

def compute_crossing_pairs(points: List[Point]):
    n = len(points)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            a = points[i]; b = points[j]
            for k in range(n):
                for l in range(k+1, n):
                    if len({i,j,k,l}) < 4:
                        continue
                    c = points[k]; d = points[l]
                    if segments_properly_intersect(a,b,c,d):
                        pairs.append(((i,j),(k,l)))
    return pairs

def solve_max_area_polygon_ilp(points: List[Point],
                               time_limit: Optional[float] = None,
                               solver_name: str = "CBC",
                               msg: bool = False) -> Tuple[List[int], float]:
    n = len(points)
    if n < 3:
        raise ValueError("Need at least 3 points")
    # coefficients for oriented area
    coeff = [[0.0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(n):
            if i == j: continue
            xj, yj = points[j]
            coeff[i][j] = xi*yj - xj*yi
    crossing_pairs = compute_crossing_pairs(points)

    prob = pulp.LpProblem("MaxAreaPolygonization", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(n), range(n)), 0, 1, cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n), 1, n, cat="Continuous")

    for i in range(n):
        prob += (x[i][i] == 0)
        prob += (pulp.lpSum(x[i][j] for j in range(n) if j != i) == 1)  # out-degree
        prob += (pulp.lpSum(x[j][i] for j in range(n) if j != i) == 1)  # in-degree

    for (i,j), (k,l) in crossing_pairs:
        prob += (x[i][j] + x[j][i] + x[k][l] + x[l][k] <= 1)

    prob += (u[0] == 1)
    for i in range(n):
        for j in range(n):
            if i == j or i == 0: 
                continue
            prob += (u[i] - u[j] + n * x[i][j] <= n - 1)

    prob += pulp.lpSum(coeff[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)

    sname = (solver_name or "CBC").upper()
    if sname == "CBC":
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=msg)
    elif sname == "GUROBI":
        solver = pulp.GUROBI_CMD(timeLimit=time_limit, msg=msg)
    elif sname == "CPLEX":
        solver = pulp.CPLEX_CMD(timeLimit=time_limit, msg=msg)
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=msg)

    prob.solve(solver)

    succ = [-1]*n
    for i in range(n):
        for j in range(n):
            val = pulp.value(x[i][j])
            if i != j and val is not None and val > 0.5:
                succ[i] = j
                break

    if any(s == -1 for s in succ):
        raise RuntimeError("No Hamiltonian cycle recovered from solver solution.")

    order = [0]
    seen = {0}
    cur = 0
    for _ in range(n-1):
        cur = succ[cur]
        if cur in seen or cur == -1:
            break
        order.append(cur)
        seen.add(cur)
    if len(order) != n or succ[cur] != order[0]:
        raise RuntimeError("Extracted tour is not a single Hamiltonian cycle.")

    # compute area
    area2 = 0.0
    for i in range(n):
        a = order[i]; b = order[(i+1) % n]
        xa, ya = points[a]; xb, yb = points[b]
        area2 += xa*yb - xb*ya
    return order, abs(area2) * 0.5

def _read_points_stdin() -> List[Point]:
    pts: List[Point] = []
    for line in sys.stdin:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        xs = s.replace(",", " ").split()
        if len(xs) < 2:
            continue
        x, y = float(xs[0]), float(xs[1])
        pts.append((x, y))
    return pts

def main():
    pts = _read_points_stdin()
    order, area = solve_max_area_polygon_ilp(pts, time_limit=None, solver_name="CBC", msg=False)
    print("# Best order (0-based indices):")
    print(" ".join(map(str, order)))
    print("# Area:")
    print(f"{area:.12f}")
    print("# Ordered coordinates:")
    for i in order:
        x, y = pts[i]
        print(f"{x} {y}")

if __name__ == "__main__":
    main()
