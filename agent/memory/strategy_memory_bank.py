"""
Strategy Memory Bank with Case-Based Reasoning (CBR)

Each agent (HFT, MFT, LFT, Allocator) has its own memory bank.

Workflow:
1. Detect market regime
2. Query memory: "给我这个regime下最好的策略"
3. Retrieve strategy parameters (warm start)
4. RL training on top of warm start
5. Update memory if performance improves
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class StrategyCase:
    """单个策略案例（存储在记忆库中）"""
    case_id: str
    regime_type: str  # 'high_risk', 'high_return', 'stable'
    agent_type: str   # 'hft', 'mft', 'lft', 'allocator'

    # 策略参数（网络权重等）
    strategy_params: Dict[str, Any]

    # 性能指标
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float

    # 市场条件（用于相似度匹配）
    market_volatility: float
    market_trend: float
    market_correlation: float

    # 元数据
    training_episodes: int
    timestamp: str
    success_score: float = 0.0

    def __post_init__(self):
        """计算综合成功分数"""
        if self.success_score == 0.0:
            self.success_score = (
                self.sharpe_ratio * 0.4 +
                (self.total_return / 100) * 0.3 +
                (1 + self.max_drawdown) * 0.1 +
                self.win_rate * 0.2
            )


class StrategyMemoryBank:
    """
    策略记忆库 - 支持所有agents的CBR

    结构：
    memory_bank/
    ├── high_risk/
    │   ├── hft/
    │   │   ├── case_001.pkl
    │   │   └── case_002.pkl
    │   ├── mft/
    │   ├── lft/
    │   └── allocator/
    ├── high_return/
    └── stable/
    """

    def __init__(self, memory_dir: str = "memory_bank"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 内存中的案例存储
        self.cases: Dict[str, Dict[str, List[StrategyCase]]] = {
            'high_risk': {'hft': [], 'mft': [], 'lft': [], 'allocator': []},
            'high_return': {'hft': [], 'mft': [], 'lft': [], 'allocator': []},
            'stable': {'hft': [], 'mft': [], 'lft': [], 'allocator': []},
        }

        # 创建目录结构
        for regime in self.cases.keys():
            for agent in self.cases[regime].keys():
                (self.memory_dir / regime / agent).mkdir(parents=True, exist_ok=True)

        # 加载已有案例
        self.load_all_cases()

        # Manifest 管理（版本记录 + 激活策略）
        self.manifests: Dict[str, Dict[str, Dict[str, Any]]] = {
            regime: {agent: {} for agent in self.cases[regime]} for regime in self.cases
        }
        self._load_all_manifests()

        print(f"🧠 Strategy Memory Bank initialized: {memory_dir}")
        self._print_inventory()

    def retrieve_best_strategy(
        self,
        regime_type: str,
        agent_type: str,
        market_conditions: Optional[Dict[str, float]] = None,
        top_k: int = 1
    ) -> List[StrategyCase]:
        """
        检索最佳策略（CBR的RETRIEVE步骤）

        Args:
            regime_type: 市场周期类型
            agent_type: agent类型
            market_conditions: 当前市场条件（用于相似度计算）
            top_k: 返回top-k个策略

        Returns:
            最佳策略列表
        """
        if regime_type not in self.cases or agent_type not in self.cases[regime_type]:
            return []

        cases = self.cases[regime_type][agent_type]

        if not cases:
            print(f"   📭 No memory for {agent_type} in {regime_type} regime")
            return []

        active_case_id = self.manifests.get(regime_type, {}).get(agent_type, {}).get('active')

        # 如果没有市场条件，直接按成功分数排序
        if market_conditions is None:
            ranked = sorted(cases, key=lambda c: c.success_score, reverse=True)
            best = ranked[:top_k]
            if active_case_id:
                active_case = next((c for c in cases if c.case_id == active_case_id), None)
                if active_case and active_case not in best:
                    best.insert(0, active_case)
                elif active_case and active_case in best:
                    idx = best.index(active_case)
                    best.insert(0, best.pop(idx))

            print(f"   🔍 Retrieved {len(best)} strategies for {agent_type}")
            if best:
                print(f"      Best: Sharpe={best[0].sharpe_ratio:.2f}, "
                      f"Return={best[0].total_return:.1f}%, Score={best[0].success_score:.3f}")

            return best

        # 计算相似度分数
        scored_cases = []
        for case in cases:
            similarity = self._calculate_similarity(case, market_conditions)
            combined_score = 0.6 * case.success_score + 0.4 * similarity
            if case.case_id == active_case_id:
                combined_score += 0.15  # 优先使用激活版本
            scored_cases.append((combined_score, case))

        scored_cases.sort(key=lambda x: x[0], reverse=True)
        best = [case for _, case in scored_cases[:top_k]]

        print(f"   🔍 Retrieved {len(best)} strategies for {agent_type} (similarity-weighted)")
        if best:
            print(f"      Best: Sharpe={best[0].sharpe_ratio:.2f}, "
                  f"Return={best[0].total_return:.1f}%, Score={best[0].success_score:.3f}")

        return best

    def warm_start_from_memory(
        self,
        regime_type: str,
        agent_type: str,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        从记忆中获取warm start参数

        Returns:
            策略参数字典，如果没有记忆则返回None
        """
        cases = self.retrieve_best_strategy(regime_type, agent_type, market_conditions, top_k=1)

        if not cases:
            print(f"   🆕 {agent_type}: Training from scratch (no memory)")
            return None

        best = cases[0]
        print(f"   ♻️  {agent_type}: Warm start from {best.case_id}")
        print(f"      Previous: Sharpe={best.sharpe_ratio:.2f}, Return={best.total_return:.1f}%")

        return best.strategy_params

    def store_strategy(
        self,
        regime_type: str,
        agent_type: str,
        strategy_params: Dict[str, Any],
        performance_metrics: Dict[str, float],
        market_conditions: Dict[str, float],
        training_episodes: int
    ) -> StrategyCase:
        """
        存储新策略到记忆库（CBR的RETAIN步骤）
        """
        case_id = f"{regime_type}_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        case = StrategyCase(
            case_id=case_id,
            regime_type=regime_type,
            agent_type=agent_type,
            strategy_params=strategy_params,
            sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
            total_return=performance_metrics.get('total_return', 0.0),
            max_drawdown=performance_metrics.get('max_drawdown', 0.0),
            win_rate=performance_metrics.get('win_rate', 0.5),
            market_volatility=market_conditions.get('volatility', 0.0),
            market_trend=market_conditions.get('trend', 0.0),
            market_correlation=market_conditions.get('correlation', 0.0),
            training_episodes=training_episodes,
            timestamp=datetime.now().isoformat()
        )

        # 添加到内存
        self.cases[regime_type][agent_type].append(case)

        # 保存到磁盘并更新 manifest
        self._save_case(case)
        self._update_manifest(regime_type, agent_type, case)

        # 保持记忆库大小（每个agent/regime只保留top N）
        self._prune_memory(regime_type, agent_type, max_cases=10)

        print(f"   💾 Stored {agent_type} strategy: Score={case.success_score:.3f}")

        return case

    def warm_start_all_agents(
        self,
        regime_type: str,
        market_conditions: Dict[str, float]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        为所有agents获取warm start参数

        Returns:
            {
                'hft': params or None,
                'mft': params or None,
                'lft': params or None,
                'allocator': params or None
            }
        """
        print(f"\n🔍 Retrieving warm start strategies for {regime_type} regime...")

        warm_starts = {}
        for agent_type in ['hft', 'mft', 'lft', 'allocator']:
            warm_starts[agent_type] = self.warm_start_from_memory(
                regime_type, agent_type, market_conditions
            )

        # 统计
        found = sum(1 for v in warm_starts.values() if v is not None)
        total = len(warm_starts)

        print(f"\n📊 Warm start summary: {found}/{total} agents have memory")

        return warm_starts

    def _calculate_similarity(
        self,
        case: StrategyCase,
        current_conditions: Dict[str, float]
    ) -> float:
        """计算市场条件相似度"""
        case_cond = {
            'volatility': case.market_volatility,
            'trend': case.market_trend,
            'correlation': case.market_correlation
        }

        distances = []
        for key in case_cond:
            if key in current_conditions:
                c_val = case_cond[key]
                curr_val = current_conditions[key]
                norm_dist = abs(c_val - curr_val) / (abs(c_val) + abs(curr_val) + 1e-8)
                distances.append(norm_dist)

        if not distances:
            return 0.5

        avg_distance = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_distance)

        return similarity

    def _prune_memory(self, regime_type: str, agent_type: str, max_cases: int = 10):
        """保持记忆库大小"""
        cases = self.cases[regime_type][agent_type]

        if len(cases) > max_cases:
            cases.sort(key=lambda c: c.success_score, reverse=True)
            removed = cases[max_cases:]
            self.cases[regime_type][agent_type] = cases[:max_cases]

            removed_ids = {case.case_id for case in removed}

            for case in removed:
                case_file = self.memory_dir / regime_type / agent_type / f"{case.case_id}.pkl"
                if case_file.exists():
                    case_file.unlink()
                json_file = self.memory_dir / regime_type / agent_type / f"{case.case_id}.json"
                if json_file.exists():
                    json_file.unlink()

            manifest = self.manifests.get(regime_type, {}).get(agent_type)
            if manifest and manifest.get('versions'):
                manifest['versions'] = [v for v in manifest['versions'] if v['case_id'] not in removed_ids]
                if manifest.get('active') in removed_ids:
                    manifest['active'] = manifest['versions'][-1]['case_id'] if manifest['versions'] else None
                self._write_manifest(regime_type, agent_type)

    def _save_case(self, case: StrategyCase):
        """保存案例到磁盘"""
        case_dir = self.memory_dir / case.regime_type / case.agent_type
        case_dir.mkdir(parents=True, exist_ok=True)

        # 保存pickle
        pkl_file = case_dir / f"{case.case_id}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(case, f)

        # 保存JSON（可读）
        json_file = case_dir / f"{case.case_id}.json"
        case_dict = asdict(case)
        # 移除可能很大的参数字典
        case_dict['strategy_params'] = "<saved in .pkl file>"
        with open(json_file, 'w') as f:
            json.dump(case_dict, f, indent=2)

    def load_all_cases(self):
        """从磁盘加载所有案例"""
        loaded = 0

        for regime in self.cases.keys():
            for agent in self.cases[regime].keys():
                case_dir = self.memory_dir / regime / agent

                if not case_dir.exists():
                    continue

                for pkl_file in case_dir.glob("*.pkl"):
                    try:
                        with open(pkl_file, 'rb') as f:
                            case = pickle.load(f)
                        self.cases[regime][agent].append(case)
                        loaded += 1
                    except Exception as e:
                        print(f"Warning: Failed to load {pkl_file}: {e}")

        if loaded > 0:
            print(f"   📚 Loaded {loaded} existing cases from disk")

    def _print_inventory(self):
        """打印记忆库库存"""
        total = sum(
            len(self.cases[regime][agent])
            for regime in self.cases
            for agent in self.cases[regime]
        )

        if total == 0:
            print("   📭 Memory bank is empty (will learn from scratch)")
            return

        print(f"   📊 Total: {total} strategy cases in memory")

        for regime in ['high_risk', 'high_return', 'stable']:
            regime_total = sum(len(self.cases[regime][agent]) for agent in self.cases[regime])
            if regime_total > 0:
                print(f"      {regime}: {regime_total} cases", end="")

                # 显示每个agent的数量
                agent_counts = []
                for agent in ['hft', 'mft', 'lft', 'allocator']:
                    count = len(self.cases[regime][agent])
                    if count > 0:
                        marker = "*" if self.manifests.get(regime, {}).get(agent, {}).get('active') else ""
                        agent_counts.append(f"{agent}:{count}{marker}")

                if agent_counts:
                    print(f" ({', '.join(agent_counts)})")
                else:
                    print()

    # Manifest helpers -------------------------------------------------

    def _manifest_path(self, regime: str, agent: str) -> Path:
        return self.memory_dir / regime / agent / "manifest.json"

    def _load_all_manifests(self):
        for regime in self.cases:
            for agent in self.cases[regime]:
                self.manifests[regime][agent] = self._load_manifest(regime, agent)

    def _load_manifest(self, regime: str, agent: str) -> Dict[str, Any]:
        manifest_path = self._manifest_path(regime, agent)
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                if 'versions' not in manifest:
                    manifest['versions'] = []
            except Exception:
                manifest = {'versions': [], 'active': None}
        else:
            manifest = {'versions': [], 'active': None}

        if not manifest['versions'] and self.cases[regime][agent]:
            sorted_cases = sorted(self.cases[regime][agent], key=lambda c: c.timestamp)
            for idx, case in enumerate(sorted_cases, start=1):
                manifest['versions'].append(self._manifest_entry(case, idx))
            manifest['active'] = manifest['versions'][-1]['case_id'] if manifest['versions'] else None
            self._write_manifest(regime, agent, manifest)

        if manifest.get('active') is None and manifest['versions']:
            manifest['active'] = manifest['versions'][-1]['case_id']
            self._write_manifest(regime, agent, manifest)

        return manifest

    def _write_manifest(self, regime: str, agent: str, manifest: Optional[Dict[str, Any]] = None):
        if manifest is not None:
            self.manifests[regime][agent] = manifest
        manifest_to_write = self.manifests[regime][agent]
        path = self._manifest_path(regime, agent)
        with open(path, 'w') as f:
            json.dump(manifest_to_write, f, indent=2)

    def _manifest_entry(self, case: StrategyCase, version: int) -> Dict[str, Any]:
        return {
            'version': version,
            'case_id': case.case_id,
            'timestamp': case.timestamp,
            'sharpe': case.sharpe_ratio,
            'total_return': case.total_return,
            'max_drawdown': case.max_drawdown,
            'win_rate': case.win_rate,
        }

    def _update_manifest(self, regime: str, agent: str, case: StrategyCase):
        manifest = self.manifests.get(regime, {}).get(agent)
        if manifest is None:
            manifest = {'versions': [], 'active': None}
            if regime not in self.manifests:
                self.manifests[regime] = {}
            self.manifests[regime][agent] = manifest

        version_no = len(manifest['versions']) + 1
        manifest['versions'].append(self._manifest_entry(case, version_no))
        manifest['active'] = case.case_id
        self._write_manifest(regime, agent)

    # Public manifest API ----------------------------------------------

    def list_versions(self, regime: str, agent: str) -> Dict[str, Any]:
        return self.manifests.get(regime, {}).get(agent, {'versions': [], 'active': None})

    def activate_case(self, regime: str, agent: str, case_id: str) -> bool:
        manifest = self.manifests.get(regime, {}).get(agent)
        if not manifest:
            return False
        if case_id not in {v['case_id'] for v in manifest['versions']}:
            return False
        manifest['active'] = case_id
        self._write_manifest(regime, agent)
        return True
