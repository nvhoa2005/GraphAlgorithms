"""
Greedy BFS solver — phiên bản tối ưu cho Online MAPD.

Nguyên lý
---------
Mỗi tick, với *từng* shipper, đánh giá **mọi ứng viên** hành động khả thi
(giao một đơn trong bag, hoặc nhặt một đơn chưa nhặt) và chấm điểm:

    score = expected_net_reward(action) / (1 + travel_distance)

Trong đó:
- expected_net_reward dùng đúng hàm `delivery_reward` của env (đã bao gồm
  bonus on-time và factor late) tính tại thời điểm dự kiến giao xong.
- travel_distance là khoảng cách BFS thực trên grid có vật cản, được
  **precompute một lần** ngay khi khởi tạo solver — biến mọi truy vấn
  khoảng cách / first-move thành tra cứu O(1).
- Trừ luôn chi phí di chuyển `move_cost(weight, W_max)` để cân nhắc tải
  trọng (đơn nặng làm chậm shipper).

Sau khi có ma trận (shipper, ứng viên) → điểm, ta dùng **Hungarian-greedy
assignment**: chọn cặp điểm cao nhất trước, gán cho tới khi không cặp nào
còn khả dụng (tránh thiên vị shipper id nhỏ như baseline cũ).

Các tối ưu khác so với baseline ban đầu
---------------------------------------
1. **Precompute BFS all-pairs** → O(M²) một lần, sau đó tra cứu O(1).
   Baseline cũ gọi BFS theo từng query trong khi chạy, cực kỳ tốn thời gian
   cho config lớn.
2. **Unified scoring** giữa giao và nhặt — không cứng nhắc "có bag thì
   phải giao". Đôi khi đi nhặt thêm 1 đơn rất gần lại có lợi hơn giao
   ngay một đơn xa.
3. **Reward-aware**: dùng đúng formula của env, không chỉ Manhattan.
4. **Move-cost aware**: trừ chi phí, ưu tiên tour ngắn cho đơn nặng.
5. **Hungarian-greedy assignment** trên cặp (shipper, ứng viên) để giảm
   bias theo shipper id.
6. **Opportunistic op=2** khi shipper đang giữ đơn — env chỉ giao đơn
   nào trùng đích sau move, nên op=2 an toàn ở mọi ô.
7. **Combine move + op** cho delivery (an toàn) → tiết kiệm 1 tick / đơn.
   Pickup giữ tách (vì `pickup_best` có thể nhặt nhầm đơn nếu move bị
   chặn).

Độ phức tạp
-----------
Gọi M = số ô trống, V = số shipper, K = số đơn quan sát.
- Precompute BFS: O(M²) thời gian + O(M²) bộ nhớ. Với N ≤ 20, M ≤ ~400,
  precompute ~ 160k phép. Rất nhanh.
- Mỗi tick: O(V · K) tính score + O(V · K · log(V · K)) sort.
  Với V ~ 5, K ~ 100, mỗi tick ~ vài nghìn phép — chạy gần như instant.

Mức tối ưu
----------
**Heuristic** (greedy 1-bước). Greedy BFS không nhìn được toàn cục như VRP
hay CBS, nên không đảm bảo tối ưu. Nhưng phiên bản này khai thác thông tin
reward/deadline/cost của env nên đã *rất tốt* trong các tình huống ít xung
đột; là baseline mạnh để các thuật toán nâng cao phải vượt qua.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from env import (
    DeliveryEnv,
    Order,
    Shipper,
    delivery_reward,
    move_cost,
)
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


class GreedyBFS(Solver):
    """Greedy 1-bước có chấm điểm reward/cost, precompute BFS all-pairs."""

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        # env2.DeliveryEnv không còn public_cfg/cfg; tránh isinstance(env, env.DeliveryEnv).
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
        self.T: int = int(self.env.T)
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0]) if self.rows else 0

        # BFS all-pairs: distance + first-move.
        self._dist: Dict[Position, Dict[Position, int]] = {}
        self._step: Dict[Position, Dict[Position, Move]] = {}
        self._precompute_shortest_paths()

    # ------------------------------------------------------------------
    # Precompute BFS shortest paths (đồng nhất với các solver nâng cao).
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
            # Tái dựng first-move cho mọi đích.
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
    # Helpers cho scoring.
    # ------------------------------------------------------------------
    @staticmethod
    def _bag_weight(shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _expected_reward_at(self, order: Order, t_arrival: int) -> float:
        t = min(max(t_arrival, 0), self.T - 1)
        return delivery_reward(order, t, self.T)

    @staticmethod
    def _move_cost_est(dist: int, weight: float, w_max: float) -> float:
        if dist <= 0:
            return 0.0
        per_step = move_cost(weight, w_max)
        return -per_step * dist

    def _eval_delivery(
        self,
        shipper: Shipper,
        order: Order,
        t_now: int,
        bag_weight: float,
    ) -> Optional[Tuple[float, int]]:
        """Trả về (score, distance) cho hành động giao đơn `order`, hoặc None."""
        d = self._distance(shipper.position, (order.ex, order.ey))
        if d >= INF:
            return None
        t_arr = t_now + d + 1
        reward = self._expected_reward_at(order, t_arr)
        cost = self._move_cost_est(d, bag_weight, shipper.W_max)
        # Cho phép reward âm/thấp (đơn quá hạn) vẫn có thể được chọn nếu
        # chỉ còn nó để giao, nên không cắt sớm ở đây — chỉ trừ chi phí.
        net = reward - cost
        return net / (d + 1.0), d

    def _eval_pickup(
        self,
        shipper: Shipper,
        order: Order,
        t_now: int,
        bag_weight: float,
    ) -> Optional[Tuple[float, int]]:
        """Trả về (score, distance) cho việc đi nhặt đơn `order`, hoặc None."""
        d_p = self._distance(shipper.position, (order.sx, order.sy))
        if d_p >= INF:
            return None
        d_d = self._distance((order.sx, order.sy), (order.ex, order.ey))
        if d_d >= INF:
            return None
        t_arr = t_now + d_p + 1 + d_d + 1
        reward = self._expected_reward_at(order, t_arr)
        # Chi phí: phần đi tới pickup chở trọng lượng hiện tại + phần đi
        # tới delivery chở thêm trọng lượng đơn mới.
        cost = self._move_cost_est(d_p, bag_weight, shipper.W_max) + self._move_cost_est(
            d_d, bag_weight + order.w, shipper.W_max
        )
        net = reward - cost
        if net <= 0:
            return None
        return net / (d_p + d_d + 2.0), d_p

    # ------------------------------------------------------------------
    # Sinh danh sách cặp (score, shipper, order, op_type) — sau đó
    # Hungarian-greedy assign.
    # ------------------------------------------------------------------
    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t_now: int = int(obs.get("t", 0))

        # Tính cached bag_weight cho mỗi shipper.
        bag_w_map: Dict[int, float] = {
            s.id: self._bag_weight(s, orders) for s in shippers
        }

        # ----- Sinh tất cả cặp ứng viên -----
        candidates: List[Tuple[float, int, str, int]] = []
        # mỗi entry: (score, shipper_id, op_type ∈ {"P","D"}, order_id)

        for s in shippers:
            bw = bag_w_map[s.id]
            # Delivery candidates (đơn trong bag).
            for oid in s.bag:
                o = orders.get(oid)
                if o is None or o.delivered:
                    continue
                ev = self._eval_delivery(s, o, t_now, bw)
                if ev is None:
                    continue
                score, _ = ev
                candidates.append((score, s.id, "D", oid))

            # Pickup candidates (đơn chưa nhặt + còn slot + còn tải).
            slot_left = s.K_max - len(s.bag)
            if slot_left <= 0:
                continue
            w_left = s.W_max - bw
            for o in orders.values():
                if o.picked or o.delivered:
                    continue
                if o.w > w_left:
                    continue
                ev = self._eval_pickup(s, o, t_now, bw)
                if ev is None:
                    continue
                score, _ = ev
                candidates.append((score, s.id, "P", o.id))

        # ----- Hungarian-greedy: gán theo điểm giảm dần -----
        candidates.sort(key=lambda x: -x[0])
        assigned_shipper: Dict[int, Tuple[str, int]] = {}
        reserved_pickup: set = set()

        for _score, sid, op_t, oid in candidates:
            if sid in assigned_shipper:
                continue
            if op_t == "P" and oid in reserved_pickup:
                continue
            assigned_shipper[sid] = (op_t, oid)
            if op_t == "P":
                reserved_pickup.add(oid)

        # ----- Tạo action cụ thể cho từng shipper -----
        actions: Dict[int, Action] = {}
        for s in shippers:
            if s.id not in assigned_shipper:
                # Không có việc khả thi: vẫn cho op=2 nếu có bag (cơ hội
                # giao nếu vô tình đứng tại điểm giao của đơn nào đó).
                actions[s.id] = ("S", 2 if s.bag else 0)
                continue

            op_t, oid = assigned_shipper[s.id]
            o = orders.get(oid)
            if o is None:
                actions[s.id] = ("S", 2 if s.bag else 0)
                continue

            if op_t == "P":
                target = (o.sx, o.sy)
                if s.position == target:
                    actions[s.id] = ("S", 1)
                else:
                    move = self._next_move(s.position, target)
                    # KHÔNG combine với op=1 (nếu move bị chặn, pickup_best
                    # ở ô hiện tại có thể nhặt nhầm đơn khác). Nhưng vẫn
                    # cho op=2 nếu có bag để giao "trên đường" cơ hội.
                    actions[s.id] = (move, 2) if s.bag else (move, 0)
            else:  # "D"
                target = (o.ex, o.ey)
                if s.position == target:
                    actions[s.id] = ("S", 2)
                else:
                    move = self._next_move(s.position, target)
                    # Combine an toàn: env chỉ giao đơn trùng đích sau move.
                    actions[s.id] = (move, 2)

        return actions

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
