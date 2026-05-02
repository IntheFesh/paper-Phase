"""
Concordance Cliff Detector（C_t）。

把三个独立 cliff 估计器的信号通过 rank-based fusion 融合，
对噪声来源不同的估计器进行三角测量，给出稳健的 cliff 置信分数。

数学定义（v2 论文 §4.4）：
    C_t = (1/3) * [rank(I^(1)(t)) + rank(I^(2)(t)) + rank(I^(3)(t))]

其中 rank 是各估计器在过去 W 步窗口内的倒序相对排名：
高估计信号 → 低排名 → 低 C_t → 触发 cliff 检测（C_t < threshold）。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class ConcordanceDetector:
    """三估计器 rank-based 融合的 cliff 检测器。

    Parameters
    ----------
    window    : int   滑动窗口宽度 W（计算 rank 的历史长度），默认 16
    threshold : float C_t 低于此值时触发重规划；范围 [0, 1]，默认 0.35
    """

    def __init__(self, window: int = 16, threshold: float = 0.35):
        self.window = window
        self.threshold = threshold
        # 三个估计器的历史值（越大表示 signal 越强，即越像 cliff）
        self._hist1: deque[float] = deque(maxlen=window)
        self._hist2: deque[float] = deque(maxlen=window)
        self._hist3: deque[float] = deque(maxlen=window)
        self._concordance_history: list[float] = []

    def reset(self):
        """每个 episode 开始时调用，清空历史。"""
        self._hist1.clear()
        self._hist2.clear()
        self._hist3.clear()
        self._concordance_history.clear()

    def update(
        self,
        i1: float,   # I^(1)：β_t（大 = 更像 cliff）
        i2: float,   # I^(2)：σ² 动作方差（大 = 更像 cliff）
        i3: float,   # I^(3)：velocity curvature（大 = 更像 cliff）
    ) -> float:
        """更新历史并返回当前时刻的 concordance 分数 C_t。

        C_t 越低（接近 0）表示三个估计器一致认为当前是 cliff。
        C_t 越高（接近 1）表示处于平台内部，无需重规划。
        """
        self._hist1.append(i1)
        self._hist2.append(i2)
        self._hist3.append(i3)

        n = len(self._hist1)
        if n < 3:
            # 历史不足，保守不触发
            c_t = 1.0
        else:
            r1 = self._rank_current(self._hist1)
            r2 = self._rank_current(self._hist2)
            r3 = self._rank_current(self._hist3)
            c_t = (r1 + r2 + r3) / 3.0

        self._concordance_history.append(c_t)
        return c_t

    def is_cliff(self, c_t: Optional[float] = None) -> bool:
        """返回当前时刻是否为 cliff（C_t < threshold）。

        若不传 c_t，则使用最近一次 update() 的结果。
        """
        score = c_t if c_t is not None else (
            self._concordance_history[-1] if self._concordance_history else 1.0
        )
        return score < self.threshold

    @staticmethod
    def _rank_current(hist: deque) -> float:
        """把窗口内最后一个值归一化到 [0, 1]。

        0 = 最大值（最像 cliff），1 = 最小值（最不像 cliff）。
        高估计信号 → 在窗口内排名靠前（更极端）→ 低排名值 → 低 C_t → 触发检测。
        """
        arr = np.array(hist, dtype=np.float32)
        n = len(arr)
        if n == 1:
            return 0.5
        cur = arr[-1]
        # 计算有多少历史值 > current；current 最大时 rank = 0（最像 cliff）
        rank = np.sum(arr > cur) + 0.5 * np.sum(arr == cur)
        return float(rank) / n

    def get_stats(self) -> dict:
        """返回调试信息。"""
        if not self._concordance_history:
            return {}
        arr = np.array(self._concordance_history)
        return {
            "mean_concordance": float(arr.mean()),
            "min_concordance":  float(arr.min()),
            "n_cliff_events":   int(np.sum(arr < self.threshold)),
            "total_steps":      len(arr),
        }
