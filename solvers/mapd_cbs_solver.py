"""
Multi-Agent Pickup and Delivery với Conflict-Based Search (MAPD-CBS).

Tổng quan
---------
MAPD-CBS giải bài toán phối hợp đa tác tử bằng 2 tầng:
- **Tầng cao (High-level CBS):** duy trì một cây ràng buộc (Constraint Tree).
  Mỗi node lưu (a) tập ràng buộc cho từng agent (vertex/edge constraints),
  (b) đường đi của mỗi agent, (c) chi phí. Khi phát hiện xung đột giữa 2
  agent, node tách thành 2 con — mỗi con thêm 1 ràng buộc cho 1 agent.
- **Tầng thấp (Low-level A*):** với mỗi agent, A* thời-gian-mở-rộng tìm
  đường đi tôn trọng tất cả ràng buộc đã thêm.

Định nghĩa xung đột (theo mô hình env)
--------------------------------------
- **Vertex conflict**: hai agent đứng cùng 1 ô tại cùng thời điểm t.
- **Edge conflict (swap)**: agent i tại ô X bước sang Y, đồng thời agent j
  tại Y bước sang X (cùng thời điểm). Env xử lý va chạm bằng id-priority
  nhưng *không* xử lý edge swap → cả hai bị kẹt. CBS giải quyết dứt điểm
  trước khi gọi env.

Mô hình online
--------------
Đơn được sinh dần theo Poisson, mỗi lần "replan" CBS chỉ lên kế hoạch
trong **một cửa sổ thời gian ngắn** (`WINDOW` bước). Sau khi cửa sổ hết
hoặc có sự kiện (đơn mới / hoàn thành task / kẹt / plan bất nhất), tổng
hợp trạng thái và CBS lại.

Pipeline mỗi tick
-----------------
1. `_advance_tasks`: pop đầu hàng đợi task nếu agent đã hoàn tất
   (đã nhặt / đã giao). Phát hiện task bất nhất (bị xe khác nhặt).
2. `_needs_replan` → nếu True thì:
       a. `_assign_tasks`: gán đơn cho shipper bằng greedy theo
          score = (potential_reward) / (1 + distance). Đơn trong bag được
          xếp đầu hàng đợi theo deadline.
       b. `_cbs_solve`: chạy CBS trong cửa sổ WINDOW bước, time-bound.
3. `_action_for`: lấy bước kế tiếp từ CBS path; sinh action (move, op).

Mức tối ưu
----------
- CBS với A* admissible (heuristic BFS đúng = khoảng cách thật) là
  **complete và optimal cho MAPF** đối với hàm mục tiêu sum-of-costs khi
  chạy đến tận cùng cây. Trong implementation này ta giới hạn (a) cửa sổ
  thời gian, (b) số node CBS expand, (c) time budget — nên thực tế đạt
  **near-optimal trong cửa sổ** chứ không phải optimal toàn cục.
- Phần gán task là **heuristic**, không tối ưu — đó là phần thường gặp
  trong các hệ thống MAPD công nghiệp (token-passing, regret-insertion…).
- Vì online + heuristic assignment → toàn hệ thống là **heuristic /
  near-optimal**.

Độ phức tạp
-----------
Gọi M = số ô trống, V = số shipper, W = WINDOW, K = số đơn quan sát.
- BFS precompute: O(M^2).
- A* low-level/agent: O(M · W · log(M·W)) trường hợp xấu.
- CBS high-level: tệ nhất hàm mũ theo số xung đột; được khống chế bằng
  MAX_CBS_NODES + CBS_TIME_LIMIT_S.
- Bộ nhớ: O(V · W) cho paths, O(M^2) cho BFS table.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env import (
    ALPHA,
    DeliveryEnv,
    Order,
    Shipper,
    r_base,
)
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
TaskEntry = Tuple[str, int, Position]  # (op_type, order_id, target)

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
MOVE_OF_DELTA = {
    (-1, 0): "U",
    (1, 0): "D",
    (0, -1): "L",
    (0, 1): "R",
    (0, 0): "S",
}


class MAPDCBSSolver(Solver):
    """Online MAPD bằng CBS với task assignment + windowed planning."""

    method_name = "MAPD-CBS"

    # ----- CBS / A* hyperparameters -----
    WINDOW: int = 12          # số bước CBS lên kế hoạch mỗi lần.
    MAX_CBS_NODES: int = 60   # giới hạn số node high-level expand.
    CBS_TIME_LIMIT_S: float = 1.2
    ASTAR_NODE_LIMIT: int = 8000

    # ----- Task assignment -----
    MAX_UNPICKED_ASSIGN: int = 120

    # ----- Replan triggers -----
    REPLAN_PERIOD: int = 12
    NEW_ORDER_COOLDOWN: int = 5
    STUCK_LIMIT: int = 2

    def __init__(self, env: DeliveryEnv):
        self.env = env
        if hasattr(env, "public_cfg"):
            self.cfg = env.public_cfg
        elif hasattr(env, "cfg"):
            self.cfg = env.cfg
        else:
            self.cfg = {
                "name": getattr(env, "config_name", "unknown"),
                "N": env.N,
                "C": env.C,
                "G": env.G,
                "T": env.T,
            }
        self.grid = env.grid
        self.orders: List[Order] = []
        self.T: int = int(self.env.T)
        self.C: int = int(self.cfg["C"])
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0]) if self.rows else 0

        # BFS all-pairs distance (heuristic admissible cho A*).
        self._dist: Dict[Position, Dict[Position, int]] = {}
        self._step: Dict[Position, Dict[Position, Move]] = {}
        self._precompute_shortest_paths()

        # Hàng đợi task cho mỗi shipper.
        self.tasks: Dict[int, deque] = {i: deque() for i in range(self.C)}
        # Đường đi CBS đã hoạch định (index theo thời gian tương đối path_t0).
        self.paths: Dict[int, List[Position]] = {i: [] for i in range(self.C)}
        self.path_t0: int = -1

        self._last_replan_t: int = -(10**9)
        self._stuck_counter: Dict[int, int] = {i: 0 for i in range(self.C)}

    # ------------------------------------------------------------------
    # BFS precompute.
    # ------------------------------------------------------------------
    def _precompute_shortest_paths(self) -> None:
        free_cells: List[Position] = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0
        ]
        for start in free_cells:
            dist_map: Dict[Position, int] = {start: 0}
            step_map: Dict[Position, Move] = {start: "S"}
            parent: Dict[Position, Tuple[Position, Move]] = {}
            queue: deque[Position] = deque([start])
            while queue:
                cur = queue.popleft()
                cur_dist = dist_map[cur]
                for mv in MOVES:
                    dr, dc = DIRS[mv]
                    nr, nc = cur[0] + dr, cur[1] + dc
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                        continue
                    if self.grid[nr][nc] != 0:
                        continue
                    nxt = (nr, nc)
                    if nxt in dist_map:
                        continue
                    dist_map[nxt] = cur_dist + 1
                    parent[nxt] = (cur, mv)
                    queue.append(nxt)
            for target in dist_map:
                if target == start:
                    continue
                cur = target
                first_move = "S"
                while True:
                    prev, mv = parent[cur]
                    if prev == start:
                        first_move = mv
                        break
                    cur = prev
                step_map[target] = first_move
            self._dist[start] = dist_map
            self._step[start] = step_map

    def _distance(self, a: Position, b: Position) -> int:
        if a == b:
            return 0
        return self._dist.get(a, {}).get(b, INF)

    # ------------------------------------------------------------------
    # Task assignment (greedy + best-insertion theo end_pos hiện tại).
    # ------------------------------------------------------------------
    @staticmethod
    def _bag_weight(s: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in s.bag if oid in orders)

    def _assign_tasks(self, obs: dict) -> None:
        shippers: List[Shipper] = obs["shippers"]
        orders_map: Dict[int, Order] = obs["orders"]

        # Reset hàng đợi.
        self.tasks = {s.id: deque() for s in shippers}

        # Bước 1: xếp delivery cho các đơn đang trong bag (theo deadline).
        for s in shippers:
            in_bag = [orders_map[oid] for oid in s.bag if oid in orders_map]
            in_bag.sort(key=lambda o: (o.et, -o.p, o.id))
            for o in in_bag:
                self.tasks[s.id].append(("D", o.id, (o.ex, o.ey)))

        # Bước 2: gán đơn unpicked.
        unpicked: List[Order] = [
            o for o in orders_map.values() if (not o.delivered) and (not o.picked)
        ]
        unpicked.sort(key=lambda o: (-o.p, o.et, o.id))
        if len(unpicked) > self.MAX_UNPICKED_ASSIGN:
            unpicked = unpicked[: self.MAX_UNPICKED_ASSIGN]

        # State để theo dõi khi gán nhiều đơn liên tiếp.
        end_pos: Dict[int, Position] = {
            s.id: (self.tasks[s.id][-1][2] if self.tasks[s.id] else s.position)
            for s in shippers
        }
        bag_count: Dict[int, int] = {s.id: len(s.bag) for s in shippers}
        bag_weight: Dict[int, float] = {
            s.id: self._bag_weight(s, orders_map) for s in shippers
        }
        k_max = {s.id: s.K_max for s in shippers}
        w_max = {s.id: s.W_max for s in shippers}

        for o in unpicked:
            best_sid = -1
            best_score = -float("inf")
            for s in shippers:
                if bag_count[s.id] >= k_max[s.id]:
                    continue
                if bag_weight[s.id] + o.w > w_max[s.id]:
                    continue
                d_p = self._distance(end_pos[s.id], (o.sx, o.sy))
                if d_p >= INF:
                    continue
                d_d = self._distance((o.sx, o.sy), (o.ex, o.ey))
                if d_d >= INF:
                    continue
                potential = ALPHA[o.p] * r_base(o.w) * 2.0
                # Càng gần và càng có giá trị càng ưu tiên; phạt nhẹ những
                # đơn dài chuyến để tránh shipper bị "ràng" vào một đơn xa.
                score = potential / (d_p + d_d + 1.0)
                if score > best_score:
                    best_score = score
                    best_sid = s.id
            if best_sid < 0:
                continue
            self.tasks[best_sid].append(("P", o.id, (o.sx, o.sy)))
            self.tasks[best_sid].append(("D", o.id, (o.ex, o.ey)))
            end_pos[best_sid] = (o.ex, o.ey)
            bag_count[best_sid] += 1
            bag_weight[best_sid] += o.w

    # ------------------------------------------------------------------
    # Low-level A* với ràng buộc thời gian.
    # ------------------------------------------------------------------
    def _astar(
        self,
        start: Position,
        goal: Position,
        vertex_cons: Set[Tuple[Position, int]],
        edge_cons: Set[Tuple[Position, Position, int]],
        window: int,
    ) -> Optional[List[Position]]:
        """A* trên (pos, time). Trả về path = [(r,c)] dài tối đa window+1.

        - vertex_cons: tập {(pos, t): agent không được đứng tại pos vào time t}.
        - edge_cons: tập {(from, to, t): agent không được di chuyển from→to
          giữa time t và t+1}.
        - window: số bước tối đa.

        Quan trọng: nếu goal nằm ngoài window (> window bước), A* không thể
        đến nơi trong cửa sổ. Khi đó trả về **path tốt nhất một phần** —
        path đi đến state có khoảng cách BFS tới goal nhỏ nhất (tie-break:
        thời gian lớn hơn). Sau đó CBS sẽ replan ở cửa sổ kế.
        """
        if start == goal:
            path: List[Position] = [start]
            for t in range(1, window + 1):
                if (start, t) in vertex_cons:
                    break
                path.append(start)
            if len(path) >= window + 1:
                return path
            # Nếu không thể đứng yên đủ window thì rơi xuống A* bên dưới.

        h0 = self._distance(start, goal)
        if h0 >= INF:
            return None

        parent: Dict[Tuple[Position, int], Tuple[Position, int]] = {}
        gscore: Dict[Tuple[Position, int], int] = {(start, 0): 0}
        heap: List[Tuple[int, int, int, Position, int]] = []
        tie = 0
        heapq.heappush(heap, (h0, 0, tie, start, 0))

        nodes_expanded = 0
        goal_state: Optional[Tuple[Position, int]] = None
        # Best-partial: state có khoảng cách BFS tới goal nhỏ nhất, ưu tiên t lớn.
        best_state: Tuple[Position, int] = (start, 0)
        best_h: int = h0
        best_g: int = 0

        while heap:
            if nodes_expanded > self.ASTAR_NODE_LIMIT:
                break
            f, g, _, pos, t = heapq.heappop(heap)
            if g > gscore.get((pos, t), 10**9):
                continue
            nodes_expanded += 1

            if pos == goal:
                goal_state = (pos, t)
                break

            # Cập nhật best-partial.
            h_remaining = self._distance(pos, goal)
            if h_remaining < best_h or (h_remaining == best_h and g > best_g):
                best_h = h_remaining
                best_g = g
                best_state = (pos, t)

            if t >= window:
                continue

            for nxt_dr, nxt_dc, is_wait in (
                (0, 0, True),
                (-1, 0, False),
                (1, 0, False),
                (0, -1, False),
                (0, 1, False),
            ):
                if is_wait:
                    nxt = pos
                else:
                    nr, nc = pos[0] + nxt_dr, pos[1] + nxt_dc
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                        continue
                    if self.grid[nr][nc] != 0:
                        continue
                    nxt = (nr, nc)
                nt = t + 1
                if (nxt, nt) in vertex_cons:
                    continue
                if (pos, nxt, t) in edge_cons:
                    continue
                ng = g + 1
                if ng < gscore.get((nxt, nt), 10**9):
                    gscore[(nxt, nt)] = ng
                    parent[(nxt, nt)] = (pos, t)
                    h = self._distance(nxt, goal)
                    if h >= INF:
                        continue
                    tie += 1
                    heapq.heappush(heap, (ng + h, ng, tie, nxt, nt))

        # Chọn state cuối: goal nếu tới được, ngược lại best-partial.
        final_state = goal_state if goal_state is not None else best_state

        if final_state == (start, 0):
            # Không đi được đâu cả: cố đứng yên (nếu được).
            path = [start]
            for t in range(1, window + 1):
                if (start, t) in vertex_cons:
                    break
                path.append(start)
            return path if path else None

        # Truy ngược path tới start.
        rev: List[Position] = []
        cur = final_state
        while True:
            rev.append(cur[0])
            if cur[1] == 0:
                break
            cur = parent[cur]
        path = list(reversed(rev))

        # Mở rộng phần đuôi bằng cách đứng tại vị trí cuối nếu không bị cấm.
        last_pos = path[-1]
        while len(path) < window + 1:
            next_t = len(path)
            if (last_pos, next_t) in vertex_cons:
                break
            path.append(last_pos)

        return path

    # ------------------------------------------------------------------
    # Phát hiện xung đột.
    # ------------------------------------------------------------------
    def _find_first_conflict(
        self, paths: Dict[int, List[Position]]
    ) -> Optional[Tuple[int, int, str, dict]]:
        agent_ids = sorted(paths.keys())
        if not agent_ids:
            return None
        max_t = max(len(paths[sid]) for sid in agent_ids)
        for t in range(max_t):
            # Lấy vị trí tại thời điểm t (nếu path ngắn hơn → giữ vị trí cuối).
            pos_t: Dict[int, Position] = {
                sid: paths[sid][min(t, len(paths[sid]) - 1)] for sid in agent_ids
            }
            # Vertex conflict.
            seen: Dict[Position, int] = {}
            for sid in agent_ids:
                p = pos_t[sid]
                if p in seen:
                    return (seen[p], sid, "vertex", {"pos": p, "t": t})
                seen[p] = sid
            # Edge conflict (chỉ nếu t+1 hợp lệ).
            if t + 1 < max_t:
                pos_t1: Dict[int, Position] = {
                    sid: paths[sid][min(t + 1, len(paths[sid]) - 1)]
                    for sid in agent_ids
                }
                for i in range(len(agent_ids)):
                    ai = agent_ids[i]
                    if pos_t[ai] == pos_t1[ai]:
                        continue
                    for j in range(i + 1, len(agent_ids)):
                        aj = agent_ids[j]
                        if pos_t[aj] == pos_t1[aj]:
                            continue
                        if pos_t[ai] == pos_t1[aj] and pos_t[aj] == pos_t1[ai]:
                            return (
                                ai,
                                aj,
                                "edge",
                                {
                                    "from_a": pos_t[ai],
                                    "to_a": pos_t1[ai],
                                    "from_b": pos_t[aj],
                                    "to_b": pos_t1[aj],
                                    "t": t,
                                },
                            )
        return None

    def _count_conflicts(self, paths: Dict[int, List[Position]]) -> int:
        agent_ids = sorted(paths.keys())
        if not agent_ids:
            return 0
        max_t = max(len(paths[sid]) for sid in agent_ids)
        cnt = 0
        for t in range(max_t):
            pos_t: Dict[int, Position] = {
                sid: paths[sid][min(t, len(paths[sid]) - 1)] for sid in agent_ids
            }
            seen: Set[Position] = set()
            for sid in agent_ids:
                p = pos_t[sid]
                if p in seen:
                    cnt += 1
                seen.add(p)
            if t + 1 < max_t:
                pos_t1: Dict[int, Position] = {
                    sid: paths[sid][min(t + 1, len(paths[sid]) - 1)]
                    for sid in agent_ids
                }
                for i in range(len(agent_ids)):
                    ai = agent_ids[i]
                    for j in range(i + 1, len(agent_ids)):
                        aj = agent_ids[j]
                        if (
                            pos_t[ai] != pos_t1[ai]
                            and pos_t[aj] != pos_t1[aj]
                            and pos_t[ai] == pos_t1[aj]
                            and pos_t[aj] == pos_t1[ai]
                        ):
                            cnt += 1
        return cnt

    # ------------------------------------------------------------------
    # CBS high-level.
    # ------------------------------------------------------------------
    def _cbs_solve(self, obs: dict) -> None:
        shippers: List[Shipper] = obs["shippers"]
        t_now = int(obs.get("t", 0))

        starts: Dict[int, Position] = {s.id: s.position for s in shippers}
        goals: Dict[int, Position] = {}
        for s in shippers:
            if self.tasks[s.id]:
                _, _, target = self.tasks[s.id][0]
                goals[s.id] = target
            else:
                goals[s.id] = s.position

        # Initial paths: A* không ràng buộc.
        root_paths: Dict[int, List[Position]] = {}
        empty_v: Set[Tuple[Position, int]] = set()
        empty_e: Set[Tuple[Position, Position, int]] = set()
        for sid in starts:
            p = self._astar(
                starts[sid], goals[sid], empty_v, empty_e, self.WINDOW
            )
            if p is None or not p:
                p = [starts[sid]] * (self.WINDOW + 1)
            else:
                while len(p) < self.WINDOW + 1:
                    p.append(p[-1])
                p = p[: self.WINDOW + 1]
            root_paths[sid] = p

        # CBS search loop.
        node_id = 0
        root_cons: Dict[int, Dict[str, Set]] = {
            sid: {"vertex": set(), "edge": set()} for sid in starts
        }
        root_cost = sum(self._path_cost(p, goals[sid]) for sid, p in root_paths.items())
        open_list: List[
            Tuple[int, int, Dict[int, Dict[str, Set]], Dict[int, List[Position]]]
        ] = []
        heapq.heappush(open_list, (root_cost, node_id, root_cons, root_paths))

        cbs_start = time.time()
        best_paths = root_paths
        best_conflicts = self._count_conflicts(root_paths)
        best_cost = root_cost
        expanded = 0

        while open_list and expanded < self.MAX_CBS_NODES:
            if time.time() - cbs_start > self.CBS_TIME_LIMIT_S:
                break
            cost, _, cons, paths = heapq.heappop(open_list)

            conflict = self._find_first_conflict(paths)
            if conflict is None:
                best_paths = paths
                best_conflicts = 0
                best_cost = cost
                break

            n_conf = self._count_conflicts(paths)
            if (n_conf, cost) < (best_conflicts, best_cost):
                best_paths = paths
                best_conflicts = n_conf
                best_cost = cost

            agent_a, agent_b, ctype, cinfo = conflict
            for aid in (agent_a, agent_b):
                # Clone constraints (deep enough — sao chép set theo agent).
                new_cons: Dict[int, Dict[str, Set]] = {
                    sid: {
                        "vertex": set(cons[sid]["vertex"]),
                        "edge": set(cons[sid]["edge"]),
                    }
                    for sid in cons
                }
                if ctype == "vertex":
                    new_cons[aid]["vertex"].add((cinfo["pos"], cinfo["t"]))
                else:
                    if aid == agent_a:
                        new_cons[aid]["edge"].add(
                            (cinfo["from_a"], cinfo["to_a"], cinfo["t"])
                        )
                    else:
                        new_cons[aid]["edge"].add(
                            (cinfo["from_b"], cinfo["to_b"], cinfo["t"])
                        )

                new_path = self._astar(
                    starts[aid],
                    goals[aid],
                    new_cons[aid]["vertex"],
                    new_cons[aid]["edge"],
                    self.WINDOW,
                )
                if new_path is None or not new_path:
                    continue
                while len(new_path) < self.WINDOW + 1:
                    new_path.append(new_path[-1])
                new_path = new_path[: self.WINDOW + 1]

                new_paths = dict(paths)
                new_paths[aid] = new_path
                new_cost = sum(
                    self._path_cost(p, goals[sid]) for sid, p in new_paths.items()
                )

                node_id += 1
                heapq.heappush(open_list, (new_cost, node_id, new_cons, new_paths))
                expanded += 1

        # Commit best paths.
        for sid in starts:
            self.paths[sid] = best_paths.get(
                sid, [starts[sid]] * (self.WINDOW + 1)
            )
        self.path_t0 = t_now

    @staticmethod
    def _path_cost(path: List[Position], goal: Position) -> int:
        """Sum-of-costs: số bước cho tới khi đạt goal (chưa đạt goal => full)."""
        for i, p in enumerate(path):
            if p == goal:
                return i
        return len(path)

    # ------------------------------------------------------------------
    # Quản lý task + xác định cần replan.
    # ------------------------------------------------------------------
    def _advance_tasks(self, obs: dict) -> bool:
        """Pop task đã hoàn thành; trả về True nếu phát hiện bất nhất."""
        orders_map: Dict[int, Order] = obs["orders"]
        invalid = False
        for s in obs["shippers"]:
            q = self.tasks[s.id]
            while q:
                op_t, oid, _target = q[0]
                o = orders_map.get(oid)
                if op_t == "P":
                    if oid in s.bag:
                        q.popleft()
                        continue
                    if o is None or o.delivered or (o.picked and o.carrier != s.id):
                        q.popleft()
                        invalid = True
                        continue
                    break
                else:
                    if o is None or o.delivered:
                        q.popleft()
                        continue
                    if oid not in s.bag:
                        q.popleft()
                        invalid = True
                        continue
                    break
        return invalid

    def _needs_replan(self, obs: dict, invalid: bool) -> bool:
        t_now = int(obs.get("t", 0))
        orders_map: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        has_unpicked = any(
            (not o.delivered) and (not o.picked) for o in orders_map.values()
        )
        has_bag = any(s.bag for s in shippers)
        if not has_unpicked and not has_bag:
            return False

        if invalid:
            return True

        # Chưa từng plan, hoặc có shipper chưa có task.
        if self.path_t0 < 0:
            return True
        for s in shippers:
            if not self.tasks[s.id] and (s.bag or has_unpicked):
                return True

        # Cửa sổ planning sắp/đã hết.
        if t_now - self.path_t0 >= self.WINDOW - 1:
            return True

        # Có đơn mới (qua cooldown).
        if obs.get("new_order_ids"):
            if t_now - self._last_replan_t >= self.NEW_ORDER_COOLDOWN:
                return True

        # Bị kẹt.
        for s in shippers:
            if self._stuck_counter.get(s.id, 0) >= self.STUCK_LIMIT and self.tasks[s.id]:
                return True

        # Đến chu kỳ replan.
        if t_now - self._last_replan_t >= self.REPLAN_PERIOD:
            return True

        return False

    # ------------------------------------------------------------------
    # Sinh action dựa trên CBS path.
    # ------------------------------------------------------------------
    def _action_for(self, shipper: Shipper, t_now: int) -> Tuple[Move, int]:
        sid = shipper.id
        if not self.tasks[sid]:
            return ("S", 2 if shipper.bag else 0)

        op_t, _oid, target = self.tasks[sid][0]

        # Tra cứu position mong đợi từ path CBS.
        idx = t_now - self.path_t0
        path = self.paths.get(sid, [])
        if not path or idx < 0 or idx >= len(path) - 1:
            # Plan hết hoặc bất nhất → fallback đi thẳng tới target qua BFS.
            if shipper.position == target:
                return ("S", 1 if op_t == "P" else 2)
            move = self._step.get(shipper.position, {}).get(target, "S")
            return (move, 2) if op_t == "D" else (move, 0)

        next_pos = path[idx + 1]
        # Nếu plan bị lệch khỏi vị trí thực tế (do va chạm hiếm gặp) → fallback BFS.
        if path[idx] != shipper.position:
            if shipper.position == target:
                return ("S", 1 if op_t == "P" else 2)
            move = self._step.get(shipper.position, {}).get(target, "S")
            return (move, 2) if op_t == "D" else (move, 0)

        delta = (next_pos[0] - shipper.position[0], next_pos[1] - shipper.position[1])
        move = MOVE_OF_DELTA.get(delta, "S")

        if next_pos == target:
            return (move, 1 if op_t == "P" else 2)
        if shipper.position == target:
            return ("S", 1 if op_t == "P" else 2)
        # Cho phép op=2 "cơ hội" trên đường nếu task hiện tại là delivery.
        return (move, 2) if op_t == "D" else (move, 0)

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            invalid = self._advance_tasks(obs)

            if self._needs_replan(obs, invalid):
                self._assign_tasks(obs)
                self._cbs_solve(obs)
                self._last_replan_t = int(obs["t"])
                self._stuck_counter = {s.id: 0 for s in obs["shippers"]}

            t_now = int(obs["t"])
            actions: Dict[int, Tuple[Move, int]] = {}
            for s in obs["shippers"]:
                actions[s.id] = self._action_for(s, t_now)

            prev_positions = {s.id: s.position for s in obs["shippers"]}
            obs, _, done, _ = self.env.step(actions)

            for s in obs["shippers"]:
                prev = prev_positions.get(s.id)
                if prev is None:
                    continue
                if s.position == prev:
                    # Cập nhật bộ đếm kẹt: chỉ tính kẹt khi chưa ở target.
                    q = self.tasks[s.id]
                    if q:
                        _op, _oid, tgt = q[0]
                        if tgt != s.position:
                            self._stuck_counter[s.id] = (
                                self._stuck_counter.get(s.id, 0) + 1
                            )
                        else:
                            self._stuck_counter[s.id] = 0
                    else:
                        self._stuck_counter[s.id] = 0
                else:
                    self._stuck_counter[s.id] = 0

            if done:
                break

        return self.env.result(
            self.method_name, elapsed_sec=time.time() - start_time
        )
