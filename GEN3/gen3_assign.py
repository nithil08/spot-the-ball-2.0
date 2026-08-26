"""gen3_assign.py — pick the 24 clips subject to every constraint at once.

THE PROBLEM
  Choose 24 windows so that all of the following hold simultaneously:

    * exactly PER_COUNT (2) clips for each end-frame player count 8..19
    * exactly SITUATION_QUOTA clips of each class (5 corner / 5 gk_throw /
      5 kickoff / 9 open)
    * at most one clip per match, for source diversity
    * final-frame ball cells spread over the 16x6 grid
    * the most coherent windows available inside all of that

  Greedy fails here, and not subtly: the classes are wildly unequal in supply (corners run
  ~0.03 per match against open play's ~47), so filling counts greedily spends the scarce
  corner matches on counts that open play could have covered, and then no assignment of
  the remainder satisfies the class quota. The constraints have to be solved together.

THE MODEL
  Min-cost max-flow. A match's CLASS is fixed by the sweep, and a match supplies at most
  one clip, so the class quota can be enforced upstream of the match and the count quota
  downstream of it:

      source --quota[k]--> class k --1--> match m --1--> (m, count c) --1--> count c --2--> sink

  The (m, c) node is what makes "one clip per match" and "this window has this count"
  compose properly: the match's single unit of flow has to choose exactly one count, and
  the edge cost is the best coherence available at that count. Max flow below 24 means the
  spec is genuinely infeasible on the candidate pool, and which edge is saturated says
  why — that is worth far more than a greedy run that silently ships 21 clips.

SPREADING THE BALL, WITHOUT LYING ABOUT IT
  Cell spread is a property of the SET, not of any one clip, so it is not a linear edge
  cost and cannot go straight into the flow. It is handled by iterative reweighting:
  solve, look at which cells/rows/columns the solution over-used, add a penalty to
  candidates sitting there, and solve again. The best solution by the true objective is
  kept. That converges quickly and never trades away a hard constraint, because the hard
  constraints live in the flow network and the penalty only ever moves cost around.
"""
import collections

# ── min-cost max-flow (successive shortest paths, SPFA) ─────────────────────────
# Written out rather than imported: this repo has neither scipy nor networkx, and adding a
# dependency to a research checkout that already renders reproducibly is a worse trade than
# eighty lines of standard algorithm.
class MCMF:
    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add(self, u, v, cap, cost):
        self.g[u].append([v, cap, cost, len(self.g[v])])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1])

    def run(self, s, t):
        flow = cost = 0
        INF = float("inf")
        while True:
            dist = [INF] * self.n
            inq = [False] * self.n
            prev = [None] * self.n
            dist[s] = 0
            q = collections.deque([s])
            while q:
                u = q.popleft()
                inq[u] = False
                for i, e in enumerate(self.g[u]):
                    v, cap, c, _ = e
                    if cap > 0 and dist[u] + c < dist[v]:
                        dist[v] = dist[u] + c
                        prev[v] = (u, i)
                        if not inq[v]:
                            inq[v] = True
                            q.append(v)
            if dist[t] == INF:
                return flow, cost
            # push one unit at a time: every capacity here is 1 or 2 and the graph is tiny,
            # so the bookkeeping of a bottleneck search is not worth it
            push = INF
            v = t
            while v != s:
                u, i = prev[v]
                push = min(push, self.g[u][i][1])
                v = u
            v = t
            while v != s:
                u, i = prev[v]
                self.g[u][i][1] -= push
                self.g[self.g[u][i][0]][self.g[u][i][3]][1] += push
                v = u
            flow += push
            cost += push * dist[t]


def _row_col(cell):
    return cell[0], int(cell[1:])


def solve(cands, counts, per_count, quota, iters=60, w_cell=900, w_row=260, w_col=150,
          verbose=True):
    """Choose the clips. `cands` is a list of dicts with match/kind/count/cell/coh.

    Returns (picks, report). picks is empty when the flow cannot reach the target size,
    and the report says which class or count could not be filled.
    """
    target = len(counts) * per_count
    # best window per (match, count) — a match contributes at most one clip anyway
    best = {}
    for c in cands:
        if c["count"] not in counts:
            continue
        k = (c["match"], c["count"])
        if k not in best or c["coh"] < best[k]["coh"]:
            best[k] = c
    pool = list(best.values())
    if not pool:
        return [], {"error": "no candidates"}

    matches = sorted({c["match"] for c in pool})
    kinds = sorted(quota)
    m_idx = {m: i for i, m in enumerate(matches)}
    c_idx = {c: i for i, c in enumerate(counts)}
    kind_of = {c["match"]: c["kind"] for c in pool}
    pairs = sorted({(c["match"], c["count"]) for c in pool})
    p_idx = {p: i for i, p in enumerate(pairs)}

    S = 0
    K0 = 1
    M0 = K0 + len(kinds)
    P0 = M0 + len(matches)
    C0 = P0 + len(pairs)
    T = C0 + len(counts)
    N = T + 1

    def build(penalty):
        f = MCMF(N)
        for i, k in enumerate(kinds):
            f.add(S, K0 + i, quota[k], 0)
        for m in matches:
            f.add(K0 + kinds.index(kind_of[m]), M0 + m_idx[m], 1, 0)
        for (m, cnt) in pairs:
            c = best[(m, cnt)]
            cost = int(round(1000 * c["coh"] + penalty(c)))
            f.add(M0 + m_idx[m], P0 + p_idx[(m, cnt)], 1, cost)
            f.add(P0 + p_idx[(m, cnt)], C0 + c_idx[cnt], 1, 0)
        for cnt in counts:
            f.add(C0 + c_idx[cnt], T, per_count, 0)
        return f

    def extract(f):
        out = []
        for (m, cnt) in pairs:
            for e in f.g[M0 + m_idx[m]]:
                if e[0] == P0 + p_idx[(m, cnt)] and e[1] == 0:
                    out.append(best[(m, cnt)])
        return out

    def spread_cost(sel):
        cell = collections.Counter(c["cell"] for c in sel)
        row = collections.Counter(_row_col(c["cell"])[0] for c in sel)
        col = collections.Counter(_row_col(c["cell"])[1] for c in sel)
        dup = sum(v - 1 for v in cell.values() if v > 1)
        rvar = sum((v - len(sel) / 6) ** 2 for v in row.values())
        cvar = sum((v - len(sel) / 16) ** 2 for v in col.values())
        return dup * 3.0 + rvar * 0.25 + cvar * 0.12

    def objective(sel):
        return sum(c["coh"] for c in sel) / len(sel) + 0.06 * spread_cost(sel)

    f = build(lambda c: 0)
    flow, _ = f.run(S, T)
    sel = extract(f)
    if flow < target:
        return [], _diagnose(pool, counts, per_count, quota, flow, target)

    best_sel, best_obj = sel, objective(sel)
    cell_u = collections.Counter(c["cell"] for c in sel)
    row_u = collections.Counter(_row_col(c["cell"])[0] for c in sel)
    col_u = collections.Counter(_row_col(c["cell"])[1] for c in sel)
    for it in range(iters):
        def penalty(c, cu=cell_u, ru=row_u, lu=col_u):
            r, l = _row_col(c["cell"])
            return (w_cell * max(0, cu[c["cell"]] - 1)
                    + w_row * max(0, ru[r] - target // 6)
                    + w_col * max(0, lu[l] - max(1, target // 16)))
        f = build(penalty)
        flow, _ = f.run(S, T)
        if flow < target:
            break
        sel = extract(f)
        obj = objective(sel)
        if obj < best_obj:
            best_sel, best_obj = sel, obj
        cell_u = collections.Counter(c["cell"] for c in sel)
        row_u = collections.Counter(_row_col(c["cell"])[0] for c in sel)
        col_u = collections.Counter(_row_col(c["cell"])[1] for c in sel)
    if verbose:
        print(f"  assignment: {len(best_sel)} clips, mean coherence "
              f"{sum(c['coh'] for c in best_sel)/len(best_sel):.3f}, "
              f"spread cost {spread_cost(best_sel):.2f}")
    return best_sel, {"objective": round(best_obj, 4)}


def _diagnose(pool, counts, per_count, quota, flow, target):
    """Say WHICH constraint the pool cannot satisfy, not just that it cannot."""
    by_kind = collections.Counter(c["kind"] for c in {(x["match"]): x for x in pool}.values())
    per_c = collections.Counter()
    for cnt in counts:
        per_c[cnt] = len({c["match"] for c in pool if c["count"] == cnt})
    return {
        "error": f"only {flow}/{target} clips assignable",
        "matches_per_class": dict(by_kind),
        "class_quota": dict(quota),
        "classes_short": {k: v for k, v in quota.items() if by_kind.get(k, 0) < v},
        "matches_per_count": dict(per_c),
        "counts_short": {c: per_c[c] for c in counts if per_c[c] < per_count},
    }
