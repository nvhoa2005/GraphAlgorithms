"""
Ant Colony Optimization (ACO) solver cho Online MAPD.

Tổng quan
---------
ACO là metaheuristic mô phỏng đàn kiến. Mỗi "kiến" xây dựng một giải pháp
(routes cho toàn bộ shipper). Lựa chọn bước kế tiếp dựa trên (a) **pheromone**
τ(i → j) (kinh nghiệm tích lũy) và (b) **heuristic** η(i → j) (kiến thức tức
thì: phần thưởng / khoảng cách). Sau mỗi vòng, pheromone bay hơi và được
bồi đắp dựa trên chất lượng các giải pháp đã sinh ra.

Trong bài toán này, "node" của ACO là **operation** chứ không phải ô lưới:
    node = ("start", shipper_id)  — vị trí bắt đầu của xe
    node = ("P", order_id)        — pickup đơn order_id
    node = ("D", order_id)        — delivery đơn order_id

Pipeline
--------
1. Precompute BFS all-pairs shortest paths trên grid (tương tự VRP).
2. Vòng lặp tick (rolling horizon):
       - Loại bỏ các bước đã hoàn tất khỏi đầu plan.
       - Nếu cần (plan trống / bất nhất / stale / có đơn mới / kẹt) thì
         gọi ACO để rebuild kế hoạch.
       - Sinh action: pickup tách biệt (an toàn), delivery kết hợp move+op=2.
3. Trong mỗi lần ACO:
       - Khởi tạo / kế thừa pheromone toàn cục τ.
       - Lặp N_ITER, mỗi vòng thả N_ANTS kiến xây route cho từng shipper.
       - Cập nhật pheromone: bay hơi + bồi đắp theo điểm của (best-iter, best-so-far).
       - Áp dụng cơ chế Min–Max ACO để τ luôn nằm trong [TAU_MIN, TAU_MAX]
         giúp tránh hội tụ sớm.
4. Có greedy fallback (cùng cấu trúc heuristic) phòng trường hợp ACO không
   ra được route hợp lệ (rất hiếm).

Tại sao có thể tốt hơn Greedy BFS?
- Greedy BFS chỉ nhìn đơn "gần nhất" / "ưu tiên cao nhất" cho mỗi shipper
  ngay thời điểm quyết định, không tính toán đa-bước.
- ACO mô phỏng đa-bước: kiến đánh giá toàn chuỗi pickup→delivery, cân nhắc
  deadline và chi phí di chuyển. Pheromone học những "chuỗi đơn nên đi liền
  nhau" sau nhiều vòng → vượt giới hạn của tham lam một-bước.
- Pheromone giữ giữa các lần replan giúp khai thác kinh nghiệm cũ.

Độ phức tạp
-----------
Gọi M = số ô trống; K = số đơn quan sát; V = số shipper.
- Precompute BFS: O(M^2).
- Mỗi ant build: O(V · K^2) tệ nhất (chọn bước kế trong O(K) ứng viên).
- Mỗi lần ACO: O(N_ITER · N_ANTS · V · K^2). Với K ~ 100, N_ITER ~ 15,
  N_ANTS ~ 10, V ~ 5: ~7.5 · 10^6 phép thao tác, < 1 giây Python.
- Bộ nhớ: O(K^2) cho pheromone (sparse dict), O(V · K) cho plans.

Mức tối ưu
----------
**Heuristic / near-optimal trong phạm vi thời gian solve**. ACO không đảm bảo
tối ưu toàn cục; tuy nhiên trên các bài VRP-tương-tự ACO thường đạt 5–15%
gần tối ưu so với exact và vượt xa các thuật toán tham lam thuần.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from env import (
    ALPHA,
    BETA,
    DeliveryEnv,
    GAMMA,
    Order,
    Shipper,
    TIME_UNIT_PER_DAY,
    delivery_reward,
    r_base,
)
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
NodeId = Tuple[str, int]  # ("start"|"P"|"D", id)
PlanStep = Tuple[int, int, str, int]  # (target_r, target_c, op_type, order_id)

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


class ACOSolver(Solver):
    """Online MAPD bằng Ant Colony Optimization + rolling-horizon."""

    method_name = "ACO"

    # ----- ACO hyperparameters -----
    N_ANTS: int = 12
    N_ITERATIONS: int = 18
    ALPHA: float = 1.0   # trọng số pheromone
    BETA: float = 4.0    # trọng số heuristic — đặt cao để bám tham lam tốt
    RHO: float = 0.15    # tốc độ bay hơi
    Q: float = 1.0       # hệ số bồi đắp
    TAU_INIT: float = 1.0
    TAU_MIN: float = 0.05
    TAU_MAX: float = 6.0
    ELITE_BONUS: float = 2.0  # nhân thêm cho best-so-far solution

    # Giới hạn thời gian build một lần (chống quá nặng cho config to)
    SOLVE_TIME_BUDGET_S: float = 1.5
    MAX_UNPICKED_FOR_SOLVE: int = 100

    # ----- Replan triggers -----
    REPLAN_PERIOD: int = 40
    NEW_ORDER_COOLDOWN: int = 6
    STUCK_LIMIT: int = 3

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.T: int = int(self.env.T)
        self.C: int = int(self.cfg["C"])
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0]) if self.rows else 0

        # BFS all-pairs
        self._dist: Dict[Position, Dict[Position, int]] = {}
        self._step: Dict[Position, Dict[Position, Move]] = {}
        self._precompute_shortest_paths()

        # Pheromone toàn cục — giữ qua các lần replan để học liên tục.
        # key = (node_from, node_to), với node = ("start", sid)/("P", oid)/("D", oid)
        self._pheromone: Dict[Tuple[NodeId, NodeId], float] = {}
        # Bộ sinh số ngẫu nhiên độc lập (không đụng RNG của env).
        self._rng = random.Random(20260520)

        self.plans: Dict[int, List[PlanStep]] = {i: [] for i in range(self.C)}
        self._last_replan_t: int = -(10**9)
        self._stuck_counter: Dict[int, int] = {i: 0 for i in range(self.C)}

    # ------------------------------------------------------------------
    # BFS precompute (đồng nhất với VRP solver).
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

    def _next_move(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        return self._step.get(a, {}).get(b, "S")

    # ------------------------------------------------------------------
    # Heuristic helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _bag_weight(shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _exp_reward_at(self, order: Order, t_arrival: int) -> float:
        """Phần thưởng dự kiến nếu giao tại t_arrival, tham chiếu hàm env."""
        t = min(max(t_arrival, 0), self.T - 1)
        return delivery_reward(order, t, self.T)

    def _move_cost_estimate(self, dist: int, weight_carried: float, w_max: float) -> float:
        """Ước lượng chi phí di chuyển (số dương, để trừ vào lợi nhuận ant)."""
        c = 0.01 * (1.0 + GAMMA * weight_carried / max(w_max, 1.0))
        return c * dist

    # ------------------------------------------------------------------
    # Pheromone helpers.
    # ------------------------------------------------------------------
    def _tau(self, frm: NodeId, to: NodeId) -> float:
        return self._pheromone.get((frm, to), self.TAU_INIT)

    def _clip_tau(self, v: float) -> float:
        return max(self.TAU_MIN, min(self.TAU_MAX, v))

    def _evaporate(self) -> None:
        keep = 1.0 - self.RHO
        for k in list(self._pheromone.keys()):
            new_v = self._pheromone[k] * keep
            if new_v < self.TAU_MIN:
                new_v = self.TAU_MIN
            self._pheromone[k] = new_v

    def _deposit(self, edges: List[Tuple[NodeId, NodeId]], amount: float) -> None:
        if amount <= 0 or not edges:
            return
        for e in edges:
            cur = self._pheromone.get(e, self.TAU_INIT)
            self._pheromone[e] = self._clip_tau(cur + amount)

    # ------------------------------------------------------------------
    # Một ant xây dựng giải pháp (routes cho mọi shipper).
    # Trả về: (routes_per_shipper, total_score, edges_used).
    # ------------------------------------------------------------------
    def _build_one_ant(
        self,
        obs: dict,
        unpicked: List[Order],
        bag_orders: Dict[int, List[Order]],
        rng: random.Random,
        greedy: bool = False,
    ) -> Tuple[Dict[int, List[PlanStep]], float, List[Tuple[NodeId, NodeId]]]:
        orders_map: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t_now: int = int(obs.get("t", 0))

        routes: Dict[int, List[PlanStep]] = {s.id: [] for s in shippers}
        edges: List[Tuple[NodeId, NodeId]] = []
        score = 0.0

        # Đơn còn rảnh để gán cho shipper khác.
        available: Dict[int, Order] = {o.id: o for o in unpicked}

        # Hoán vị thứ tự xử lý shipper để giảm bias.
        order_of_shippers = list(shippers)
        rng.shuffle(order_of_shippers)

        for s in order_of_shippers:
            pos: Position = s.position
            bag_set = set(s.bag)
            bag_w = self._bag_weight(s, orders_map)
            bag_n = len(bag_set)
            cur_time = t_now
            last_node: NodeId = ("start", s.id)

            # Vòng lặp xây route cho shipper s.
            while True:
                candidates: List[Tuple[str, int, float, int, float]] = []
                # mỗi tuple: (op_type, oid, weight_aco, dist_to_target, exp_reward)

                # ---- Ứng viên 1: giao đơn đang giữ ----
                for oid in bag_set:
                    o = orders_map.get(oid)
                    if o is None:
                        continue
                    d = self._distance(pos, (o.ex, o.ey))
                    if d >= INF:
                        continue
                    t_arrive = cur_time + d + 1
                    rwd = self._exp_reward_at(o, t_arrive)
                    cost = self._move_cost_estimate(d, bag_w, s.W_max)
                    net = rwd - cost
                    if net <= 0.0:
                        # Vẫn cần giao đơn đã nhặt; cho điểm nhỏ dương để không bỏ.
                        net = max(net, 0.05)
                    eta = net / (d + 1.0)
                    tau = self._tau(last_node, ("D", oid))
                    w_aco = (max(tau, 1e-9) ** self.ALPHA) * (max(eta, 1e-9) ** self.BETA)
                    candidates.append(("D", oid, w_aco, d, rwd))

                # ---- Ứng viên 2: nhặt đơn còn rảnh ----
                if bag_n < s.K_max:
                    for oid, o in available.items():
                        if bag_w + o.w > s.W_max:
                            continue
                        dp = self._distance(pos, (o.sx, o.sy))
                        if dp >= INF:
                            continue
                        dd = self._distance((o.sx, o.sy), (o.ex, o.ey))
                        if dd >= INF:
                            continue
                        t_arrive_delivery = cur_time + dp + 1 + dd + 1
                        rwd = self._exp_reward_at(o, t_arrive_delivery)
                        if rwd <= 0:
                            continue
                        cost = self._move_cost_estimate(dp, bag_w, s.W_max)
                        cost += self._move_cost_estimate(dd, bag_w + o.w, s.W_max)
                        net = rwd - cost
                        if net <= 0:
                            continue
                        eta = net / (dp + dd + 2.0)
                        # Urgency bonus: ưu tiên đơn deadline gần.
                        urgency = 1.0 + max(0.0, (TIME_UNIT_PER_DAY / max(o.et - cur_time, 1)) * 0.05)
                        eta *= urgency
                        tau = self._tau(last_node, ("P", oid))
                        w_aco = (max(tau, 1e-9) ** self.ALPHA) * (max(eta, 1e-9) ** self.BETA)
                        candidates.append(("P", oid, w_aco, dp, rwd))

                if not candidates:
                    break

                # ---- Lấy mẫu ứng viên ----
                if greedy:
                    op_type, oid, _, d, exp_r = max(candidates, key=lambda x: x[2])
                else:
                    total_w = sum(c[2] for c in candidates)
                    if total_w <= 0:
                        op_type, oid, _, d, exp_r = max(candidates, key=lambda x: x[2])
                    else:
                        r = rng.random() * total_w
                        cum = 0.0
                        chosen = candidates[-1]
                        for c in candidates:
                            cum += c[2]
                            if cum >= r:
                                chosen = c
                                break
                        op_type, oid, _, d, exp_r = chosen

                # ---- Cập nhật state shipper ----
                o = orders_map[oid]
                if op_type == "P":
                    routes[s.id].append((o.sx, o.sy, "P", oid))
                    edges.append((last_node, ("P", oid)))
                    cur_time += d + 1
                    pos = (o.sx, o.sy)
                    bag_set.add(oid)
                    bag_w += o.w
                    bag_n += 1
                    available.pop(oid, None)
                    # Trừ chi phí đi tới pickup.
                    score -= self._move_cost_estimate(d, bag_w - o.w, s.W_max)
                    last_node = ("P", oid)
                else:  # "D"
                    routes[s.id].append((o.ex, o.ey, "D", oid))
                    edges.append((last_node, ("D", oid)))
                    cur_time += d + 1
                    pos = (o.ex, o.ey)
                    bag_set.discard(oid)
                    bag_w -= o.w
                    bag_n -= 1
                    score += exp_r - self._move_cost_estimate(d, bag_w + o.w, s.W_max)
                    last_node = ("D", oid)

                # Cắt độ dài route hợp lý để tránh dây dài vô tận.
                if len(routes[s.id]) >= 2 * (s.K_max + len(bag_orders.get(s.id, [])) + 30):
                    break

        return routes, score, edges

    # ------------------------------------------------------------------
    # ACO main: chạy nhiều iteration, trả về best route.
    # ------------------------------------------------------------------
    def _aco_search(
        self,
        obs: dict,
        unpicked: List[Order],
        bag_orders: Dict[int, List[Order]],
    ) -> Dict[int, List[PlanStep]]:
        start = time.time()

        # Khởi tạo / refresh các tham số.
        best_routes: Optional[Dict[int, List[PlanStep]]] = None
        best_score = -float("inf")
        best_edges: List[Tuple[NodeId, NodeId]] = []

        # Bắt đầu bằng 1 ant "greedy" để có baseline tốt.
        greedy_routes, greedy_score, greedy_edges = self._build_one_ant(
            obs, unpicked, bag_orders, self._rng, greedy=True
        )
        if greedy_score > best_score:
            best_score = greedy_score
            best_routes = greedy_routes
            best_edges = greedy_edges

        # Vòng ACO.
        for it in range(self.N_ITERATIONS):
            if time.time() - start > self.SOLVE_TIME_BUDGET_S:
                break
            iter_best_score = -float("inf")
            iter_best_edges: List[Tuple[NodeId, NodeId]] = []
            iter_best_routes: Optional[Dict[int, List[PlanStep]]] = None

            for _ in range(self.N_ANTS):
                routes, sc, eds = self._build_one_ant(
                    obs, unpicked, bag_orders, self._rng, greedy=False
                )
                if sc > iter_best_score:
                    iter_best_score = sc
                    iter_best_edges = eds
                    iter_best_routes = routes
                if sc > best_score:
                    best_score = sc
                    best_routes = routes
                    best_edges = eds

            # Update pheromone.
            self._evaporate()
            # Bồi đắp theo best-of-iteration.
            if iter_best_routes is not None and iter_best_score > 0:
                norm = max(1.0, abs(best_score) if best_score > 0 else iter_best_score)
                amt = self.Q * (iter_best_score / norm)
                self._deposit(iter_best_edges, amt)
            # Bồi đắp thêm cho best-so-far (elite).
            if best_routes is not None and best_score > 0:
                norm = max(1.0, abs(best_score))
                amt = self.Q * (best_score / norm) * self.ELITE_BONUS
                self._deposit(best_edges, amt)

        return best_routes if best_routes is not None else {s.id: [] for s in obs["shippers"]}

    # ------------------------------------------------------------------
    # Replan entrypoint.
    # ------------------------------------------------------------------
    def _replan(self, obs: dict) -> None:
        orders_map: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        C = len(shippers)

        unpicked: List[Order] = []
        bag_orders: Dict[int, List[Order]] = {s.id: [] for s in shippers}
        for o in orders_map.values():
            if o.delivered:
                continue
            if o.picked and 0 <= o.carrier < C:
                bag_orders[o.carrier].append(o)
            elif not o.picked:
                unpicked.append(o)

        if not unpicked and not any(bag_orders.values()):
            for s in shippers:
                self.plans[s.id] = []
            return

        # Sắp xếp ưu tiên + cắt số đơn để ant không quá nặng.
        unpicked.sort(key=lambda o: (-o.p, o.et, o.id))
        if len(unpicked) > self.MAX_UNPICKED_FOR_SOLVE:
            unpicked = unpicked[: self.MAX_UNPICKED_FOR_SOLVE]

        routes = self._aco_search(obs, unpicked, bag_orders)
        for s in shippers:
            self.plans[s.id] = routes.get(s.id, [])

    # ------------------------------------------------------------------
    # Quản lý plan (giống VRP solver).
    # ------------------------------------------------------------------
    def _advance_plans(self, obs: dict) -> bool:
        orders_map: Dict[int, Order] = obs["orders"]
        invalid = False
        for s in obs["shippers"]:
            plan = self.plans[s.id]
            while plan:
                _tr, _tc, op_t, oid = plan[0]
                o = orders_map.get(oid)
                if op_t == "P":
                    if oid in s.bag:
                        plan.pop(0)
                        continue
                    if o is None or o.delivered or (o.picked and o.carrier != s.id):
                        plan.pop(0)
                        invalid = True
                        continue
                    break
                else:
                    if o is None or o.delivered:
                        plan.pop(0)
                        continue
                    if oid not in s.bag:
                        plan.pop(0)
                        invalid = True
                        continue
                    break
        return invalid

    def _needs_replan(self, obs: dict, invalid_after_advance: bool) -> bool:
        t_now: int = int(obs.get("t", 0))
        orders_map: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        has_unpicked = any(
            (not o.delivered) and (not o.picked) for o in orders_map.values()
        )
        has_bag = any(s.bag for s in shippers)
        if not has_unpicked and not has_bag:
            return False

        if invalid_after_advance:
            return True

        for s in shippers:
            if not self.plans[s.id]:
                if s.bag or has_unpicked:
                    return True

        if obs.get("new_order_ids"):
            if t_now - self._last_replan_t >= self.NEW_ORDER_COOLDOWN:
                return True

        for s in shippers:
            if self._stuck_counter.get(s.id, 0) >= self.STUCK_LIMIT and self.plans[s.id]:
                return True

        if t_now - self._last_replan_t >= self.REPLAN_PERIOD:
            return True

        return False

    def _action_for(self, shipper: Shipper) -> Tuple[Move, int]:
        plan = self.plans[shipper.id]
        if not plan:
            return ("S", 2 if shipper.bag else 0)
        tr, tc, op_t, _oid = plan[0]
        target = (tr, tc)
        if shipper.position == target:
            return ("S", 1 if op_t == "P" else 2)
        move = self._next_move(shipper.position, target)
        if op_t == "D":
            return (move, 2)
        return (move, 0)

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            invalid = self._advance_plans(obs)

            if self._needs_replan(obs, invalid):
                self._replan(obs)
                self._last_replan_t = int(obs["t"])
                self._stuck_counter = {s.id: 0 for s in obs["shippers"]}

            actions: Dict[int, Tuple[Move, int]] = {}
            for s in obs["shippers"]:
                actions[s.id] = self._action_for(s)

            prev_positions = {s.id: s.position for s in obs["shippers"]}
            obs, _, done, _ = self.env.step(actions)

            for s in obs["shippers"]:
                prev = prev_positions.get(s.id)
                if prev is None:
                    continue
                if s.position == prev:
                    plan = self.plans[s.id]
                    if plan:
                        tr, tc, _op, _oid = plan[0]
                        if (tr, tc) != s.position:
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
