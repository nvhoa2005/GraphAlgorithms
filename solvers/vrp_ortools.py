from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from env import (
    ALPHA,
    BETA,
    DeliveryEnv,
    Order,
    Shipper,
    is_valid_cell,
    r_base,
    valid_next_pos,
)
from solvers.solver import Solver

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    _HAS_ORTOOLS = True
except Exception:
    _HAS_ORTOOLS = False


Move = str
Position = Tuple[int, int]
# (target_r, target_c, op_type ∈ {"P","D"}, order_id)
PlanStep = Tuple[int, int, str, int]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

# Hệ số scale
REWARD_SCALE = 100
WEIGHT_SCALE = 100


class VRPOrToolsSolver(Solver):

    method_name = "VRPOrTools"

    # Số bước tối đa giữa hai lần replan dù không có sự kiện
    REPLAN_PERIOD = 40
    # Cooldown khi chỉ có sự kiện đơn mới xuất hiện
    NEW_ORDER_REPLAN_COOLDOWN = 6
    # Số bước kẹt liên tiếp tối đa trước khi replan
    STUCK_LIMIT = 3
    # Giới hạn số đơn unpicked đưa vào một lần solve
    MAX_UNPICKED_FOR_SOLVE = 80

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

        # BFS all-pairs
        self._dist: Dict[Position, Dict[Position, int]] = {}
        self._step: Dict[Position, Dict[Position, Move]] = {}
        self._precompute_shortest_paths()

        self.plans: Dict[int, List[PlanStep]] = {i: [] for i in range(self.C)}
        self._last_replan_t: int = -(10**9)
        self._last_position: Dict[int, Position] = {}
        self._stuck_counter: Dict[int, int] = {i: 0 for i in range(self.C)}

    # Precompute BFS shortest paths.
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

    # Helpers
    @staticmethod
    def _bag_weight(shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    @staticmethod
    def _potential_reward(order: Order) -> float:
        return ALPHA[order.p] * r_base(order.w) * 2.0

    @staticmethod
    def _late_floor_reward(order: Order) -> float:
        return BETA[order.p] * r_base(order.w)

    # Greedy fallback
    def _greedy_replan(
        self,
        obs: dict,
        unpicked: List[Order],
        bag_orders: Dict[int, List[Order]],
    ) -> None:
        
        shippers: List[Shipper] = obs["shippers"]
        orders: Dict[int, Order] = obs["orders"]
        reserved: set = set()

        # Đầu tiên đặt lịch giao những đơn đã trong bag (sort theo deadline)
        for s in shippers:
            self.plans[s.id] = []
            in_bag = sorted(bag_orders.get(s.id, []), key=lambda o: (o.et, -o.p, o.id))
            for o in in_bag:
                self.plans[s.id].append((o.ex, o.ey, "D", o.id))

        if not unpicked:
            return

        # Với mỗi shipper, chọn đơn unpicked có "lợi / chi phí" lớn nhất
        for s in shippers:
            current_pos = s.position
            current_bag_count = len(s.bag)
            current_bag_weight = self._bag_weight(s, orders)

            # Cho phép gán nhiều đơn liên tiếp nếu còn slot
            for _ in range(s.K_max - current_bag_count):
                best: Optional[Order] = None
                best_score = -1.0
                for o in unpicked:
                    if o.id in reserved:
                        continue
                    if current_bag_count >= s.K_max:
                        break
                    if current_bag_weight + o.w > s.W_max:
                        continue
                    d_pickup = self._distance(current_pos, (o.sx, o.sy))
                    if d_pickup >= INF:
                        continue
                    d_total = d_pickup + self._distance((o.sx, o.sy), (o.ex, o.ey))
                    if d_total >= INF:
                        continue
                    score = self._potential_reward(o) / (d_total + 1.0)
                    if score > best_score:
                        best_score = score
                        best = o
                if best is None:
                    break
                reserved.add(best.id)
                self.plans[s.id].append((best.sx, best.sy, "P", best.id))
                self.plans[s.id].append((best.ex, best.ey, "D", best.id))
                current_pos = (best.ex, best.ey)
                current_bag_count += 1
                current_bag_weight += best.w

    # VRP replan với OR-Tools
    def _replan(self, obs: dict) -> None:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t_now: int = int(obs.get("t", 0))
        C = len(shippers)

        unpicked: List[Order] = []
        bag_orders: Dict[int, List[Order]] = {s.id: [] for s in shippers}
        for o in orders.values():
            if o.delivered:
                continue
            if o.picked and 0 <= o.carrier < C:
                bag_orders[o.carrier].append(o)
            elif not o.picked:
                unpicked.append(o)

        has_bag = any(bag_orders[s.id] for s in shippers)
        if not unpicked and not has_bag:
            for s in shippers:
                self.plans[s.id] = []
            return

        # Sắp xếp ưu tiên: priority cao trước, deadline gần trước
        unpicked.sort(key=lambda o: (-o.p, o.et, o.id))
        if len(unpicked) > self.MAX_UNPICKED_FOR_SOLVE:
            unpicked = unpicked[: self.MAX_UNPICKED_FOR_SOLVE]
        
        if not _HAS_ORTOOLS:
            self._greedy_replan(obs, unpicked, bag_orders)
            return

        locations: List[Position] = []
        for s in shippers:
            locations.append(s.position)
        for s in shippers:
            locations.append(s.position)

        unpicked_start = 2 * C
        for o in unpicked:
            locations.append((o.sx, o.sy))
            locations.append((o.ex, o.ey))

        bag_start = len(locations)
        bag_entries: List[Tuple[int, Order]] = []
        for sid in sorted(bag_orders.keys()):
            for o in sorted(bag_orders[sid], key=lambda x: (x.et, -x.p, x.id)):
                bag_entries.append((sid, o))
                locations.append((o.ex, o.ey))

        n_nodes = len(locations)

        starts = list(range(C))
        ends = list(range(C, 2 * C))

        manager = pywrapcp.RoutingIndexManager(n_nodes, C, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # Distance / arc cost
        def transit_cb(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Đi tới end node ảo: miễn phí.
            if C <= to_node < 2 * C:
                return 0
            return self._distance(locations[from_node], locations[to_node])

        transit_idx = routing.RegisterTransitCallback(transit_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Time dimension
        horizon = max(self.T + 50, t_now + 200)

        def time_cb(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if C <= to_node < 2 * C:
                return 0
            d = self._distance(locations[from_node], locations[to_node])
            return d + (1 if to_node >= 2 * C else 0)

        time_cb_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(time_cb_idx, 0, horizon, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        for v in range(C):
            time_dim.CumulVar(routing.Start(v)).SetValue(t_now)

        def _deadline_coeff(o: Order) -> int:
            base = ALPHA[o.p] * r_base(o.w)
            return max(1, int(base * REWARD_SCALE / max(self.T, 1)) + 2)

        for i, o in enumerate(unpicked):
            d_node = unpicked_start + 2 * i + 1
            d_idx = manager.NodeToIndex(d_node)
            ub = max(t_now, min(o.et, horizon - 1))
            time_dim.SetCumulVarSoftUpperBound(d_idx, ub, _deadline_coeff(o))

        for offset, (_sid, o) in enumerate(bag_entries):
            d_node = bag_start + offset
            d_idx = manager.NodeToIndex(d_node)
            ub = max(t_now, min(o.et, horizon - 1))
            time_dim.SetCumulVarSoftUpperBound(d_idx, ub, _deadline_coeff(o))

        # Count capacity (số đơn trong bag)
        def count_cb(from_index: int) -> int:
            n = manager.IndexToNode(from_index)
            if n < 2 * C:
                return 0
            if n < bag_start:
                rel = n - unpicked_start
                return 1 if rel % 2 == 0 else -1
            return -1

        count_cb_idx = routing.RegisterUnaryTransitCallback(count_cb)
        routing.AddDimensionWithVehicleCapacity(
            count_cb_idx,
            0,
            [int(s.K_max) for s in shippers],
            False,
            "Count",
        )
        count_dim = routing.GetDimensionOrDie("Count")
        for v, s in enumerate(shippers):
            count_dim.CumulVar(routing.Start(v)).SetValue(len(s.bag))

        # Weight capacity
        def weight_cb(from_index: int) -> int:
            n = manager.IndexToNode(from_index)
            if n < 2 * C:
                return 0
            if n < bag_start:
                rel = n - unpicked_start
                o = unpicked[rel // 2]
                w = int(round(o.w * WEIGHT_SCALE))
                return w if rel % 2 == 0 else -w
            offset = n - bag_start
            _, o = bag_entries[offset]
            return -int(round(o.w * WEIGHT_SCALE))

        weight_cb_idx = routing.RegisterUnaryTransitCallback(weight_cb)
        routing.AddDimensionWithVehicleCapacity(
            weight_cb_idx,
            0,
            [int(round(s.W_max * WEIGHT_SCALE)) for s in shippers],
            False,
            "Weight",
        )
        weight_dim = routing.GetDimensionOrDie("Weight")
        for v, s in enumerate(shippers):
            w_bag_int = int(round(self._bag_weight(s, orders) * WEIGHT_SCALE))
            weight_dim.CumulVar(routing.Start(v)).SetValue(w_bag_int)

        # Pickup-Delivery + Disjunction
        solver = routing.solver()
        for i, o in enumerate(unpicked):
            p_node = unpicked_start + 2 * i
            d_node = p_node + 1
            p_idx = manager.NodeToIndex(p_node)
            d_idx = manager.NodeToIndex(d_node)

            routing.AddPickupAndDelivery(p_idx, d_idx)
            solver.Add(routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx))
            solver.Add(time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx))

            potential = int(round(self._potential_reward(o) * REWARD_SCALE))
            routing.AddDisjunction([p_idx], max(potential, 1))
            routing.AddDisjunction([d_idx], max(potential, 1))

        # Bag delivery: ép thuộc carrier hiện tại
        for offset, (sid, o) in enumerate(bag_entries):
            d_node = bag_start + offset
            d_idx = manager.NodeToIndex(d_node)
            routing.VehicleVar(d_idx).SetValues([sid])
            # Bỏ đơn đang ôm trên tay => phạt rất nặng.
            potential = int(round(self._potential_reward(o) * REWARD_SCALE)) * 10
            routing.AddDisjunction([d_idx], max(potential, 1))

        # Search params (time budget thích nghi)
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        n_orders = len(unpicked) + len(bag_entries)
        if n_orders <= 4:
            time_limit_ms = 150
        elif n_orders <= 10:
            time_limit_ms = 400
        elif n_orders <= 20:
            time_limit_ms = 800
        elif n_orders <= 40:
            time_limit_ms = 1300
        elif n_orders <= 70:
            time_limit_ms = 1800
        else:
            time_limit_ms = 2300
        params.time_limit.seconds = time_limit_ms // 1000
        params.time_limit.nanos = (time_limit_ms % 1000) * 1_000_000
        # Dừng sớm khi không cải tiến.
        try:
            params.lns_time_limit.seconds = 0
            params.lns_time_limit.nanos = 100_000_000
        except Exception:
            pass

        solution = routing.SolveWithParameters(params)
        if solution is None:
            # OR-Tools không tìm được nghiệm thì dùng greedy
            self._greedy_replan(obs, unpicked, bag_orders)
            return

        # Trích xuất plan
        for v in range(C):
            new_plan: List[PlanStep] = []
            idx = routing.Start(v)
            while not routing.IsEnd(idx):
                idx = solution.Value(routing.NextVar(idx))
                node = manager.IndexToNode(idx)
                if C <= node < 2 * C:
                    break
                if node < bag_start:
                    rel = node - unpicked_start
                    o = unpicked[rel // 2]
                    if rel % 2 == 0:
                        new_plan.append((o.sx, o.sy, "P", o.id))
                    else:
                        new_plan.append((o.ex, o.ey, "D", o.id))
                else:
                    offset = node - bag_start
                    _, o = bag_entries[offset]
                    new_plan.append((o.ex, o.ey, "D", o.id))
            self.plans[v] = new_plan

    # Pop những bước đã hoàn thành hoặc không còn hợp lệ
    def _advance_plans(self, obs: dict) -> bool:
        
        orders: Dict[int, Order] = obs["orders"]
        invalid = False

        for s in obs["shippers"]:
            plan = self.plans[s.id]
            while plan:
                _tr, _tc, op_t, oid = plan[0]
                o = orders.get(oid)
                if op_t == "P":
                    if oid in s.bag:
                        # Đã nhặt thành công
                        plan.pop(0)
                        continue
                    if o is None or o.delivered or (o.picked and o.carrier != s.id):
                        # Đơn không còn hoặc bị xe khác nhặt
                        plan.pop(0)
                        invalid = True
                        continue
                    break
                else:  # "D"
                    if o is None or o.delivered:
                        plan.pop(0)
                        continue
                    if oid not in s.bag:
                        # Đơn đáng lẽ trong bag nhưng không có → bất nhất
                        plan.pop(0)
                        invalid = True
                        continue
                    break

        return invalid
    
    # Quyết định có replan hay không
    def _needs_replan(self, obs: dict, invalid_after_advance: bool) -> bool:
        t_now: int = int(obs.get("t", 0))
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        has_unpicked = any(
            (not o.delivered) and (not o.picked) for o in orders.values()
        )
        has_bag = any(s.bag for s in shippers)
        if not has_unpicked and not has_bag:
            return False

        if invalid_after_advance:
            return True

        # Bất kỳ shipper nào trống lịch mà còn việc khả thi
        for s in shippers:
            if not self.plans[s.id]:
                # Còn unpicked đơn shipper có thể chở, hoặc đang giữ đơn
                if s.bag:
                    return True
                if has_unpicked:
                    return True

        # Có đơn mới (đã qua cooldown)
        if obs.get("new_order_ids"):
            if t_now - self._last_replan_t >= self.NEW_ORDER_REPLAN_COOLDOWN:
                return True

        # Có shipper kẹt nhiều bước
        for s in shippers:
            if self._stuck_counter.get(s.id, 0) >= self.STUCK_LIMIT and self.plans[s.id]:
                return True

        # Plan stale
        if t_now - self._last_replan_t >= self.REPLAN_PERIOD:
            return True

        return False

    # Sinh action cho một shipper dựa trên plan đầu tiên
    def _action_for(self, shipper: Shipper) -> Tuple[Move, int]:
        plan = self.plans[shipper.id]
        if not plan:
            return ("S", 2 if shipper.bag else 0)

        target_r, target_c, op_t, _oid = plan[0]
        target: Position = (target_r, target_c)

        if shipper.position == target:
            # Đã ở đúng ô thì thực hiện op tại chỗ
            return ("S", 1 if op_t == "P" else 2)

        move = self._next_move(shipper.position, target)
        if op_t == "D":
            # op=2 sẽ chỉ giao nếu trùng đích sau bước
            return (move, 2)

        return (move, 0)

    # Main loop
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            # Cập nhật kế hoạch dựa trên trạng thái mới nhất
            invalid = self._advance_plans(obs)

            # Replan nếu cần.
            if self._needs_replan(obs, invalid):
                self._replan(obs)
                self._last_replan_t = int(obs["t"])
                # Reset stuck counter sau replan
                self._stuck_counter = {s.id: 0 for s in obs["shippers"]}

            # Sinh action
            actions: Dict[int, Tuple[Move, int]] = {}
            for s in obs["shippers"]:
                actions[s.id] = self._action_for(s)

            # Step môi trường
            prev_positions = {s.id: s.position for s in obs["shippers"]}
            obs, _, done, _ = self.env.step(actions)

            # Cập nhật stuck-counter (shipper không di chuyển dù không phải đang đứng ở target để pickup/deliver)
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
