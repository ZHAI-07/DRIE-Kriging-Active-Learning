"""数据加载与实验分区模块

功能：
- 加载DRIE和CFD实验数据（不可变快照）
- 来源分组（2/13共享测量源，保证不跨train/test）
- 生成训练/候选池/测试集分割（防止标签泄漏）
- 标签oracle（仅按需揭示候选点标签）

数学保证：
- 来源阻断：2/13必在同侧（train或test）
- 拒绝采样：最多10000次尝试满足分组约束
- 预算严格：n_test + n_initial + budget <= 总实验数
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np


@dataclass
class Dataset:
    """数据容器（不可变）
    
    24个DRIE实验 -> 4维输入(X)，12维输出(Y)，6结构×2指标
    CFD仿真 -> 2维输入，1维输出
    """
    name: str
    X: np.ndarray
    Y: np.ndarray
    ids: np.ndarray
    feature_names: list
    target_names: list
    structures: list
    metrics: list
    source_path: str
    source_sha256: str
    provenance: dict

    def subset(self, keep):
        """按索引掩码提取子集（用于排除特定实验ID）"""
        return Dataset(self.name, self.X[keep], self.Y[keep], self.ids[keep],
                       self.feature_names, self.target_names, self.structures,
                       self.metrics, self.source_path, self.source_sha256, self.provenance)


def load_dataset(root, name, exclude_ids=(), cfd_variant="excel"):
    """从JSON快照加载数据集
    
    参数：
        name: "real"(DRIE) 或 "cfd"
        exclude_ids: 敏感性分析中排除的实验ID（如24号）
        cfd_variant: "excel"(工作簿值) 或 "legacy_code"(历史代码值)
    
    返回：
        Dataset对象，SHA256哈希固定输入来源
    """
    path = Path(root) / "data" / "raw" / (
        "user_confirmed_table_2026-11-09.json" if name == "real" else "cfd_legacy_snapshot.json")
    content = path.read_bytes()
    raw = json.loads(content.decode("utf-8-sig"))
    rows = np.asarray(raw["rows"], float)
    provenance = dict(raw.get("provenance", raw.get("source", {})))
    
    # DRIE实验：4输入(Coil/Platen/SF6/Pressure) + 12输出(6结构×2指标)
    if name == "real":
        X, Y = rows[:, 1:5], rows[:, 5:]
        labels = [s["label"] for s in raw["structures"]]
        structures = [s for s in labels for _ in range(2)]
        metrics = [m for _ in labels for m in ("CD Loss", "AR")]
        names = [f"{s} {m}" for s, m in zip(structures, metrics)]
        feature_names = ["Coil", "Platen", "SF6", "Passivation pressure"]
    # CFD仿真：2输入(Pressure/Power) + 1输出
    else:
        if cfd_variant not in {"excel", "legacy_code"}:
            raise ValueError(cfd_variant)
        # 历史代码与Excel工作簿存在数值差异，此处可选择版本
        if cfd_variant == "legacy_code":
            for difference in provenance["legacy_code_value_differences"]:
                index = difference["excel_row"] - 2
                if rows[index, 3] != difference["workbook_value"]:
                    raise ValueError("CFD discrepancy mapping no longer matches snapshot")
                rows[index, 3] = difference["code_value"]
        provenance["numerical_variant_used"] = cfd_variant
        provenance["snapshot_modified_on_disk"] = False
        X, Y = rows[:, 1:3], rows[:, 3:4]
        structures, metrics, names = ["CFD"], ["simulated response"], ["CFD response"]
        feature_names = ["Pressure", "Power"]
    
    # 检查数据完整性：无缺失值，实验ID唯一
    if not np.isfinite(rows).all() or len(set(rows[:, 0])) != len(rows):
        raise ValueError("Input contains missing values or duplicate experiment IDs")
    result = Dataset(name, X, Y, rows[:, 0].astype(int), feature_names, names,
                     structures, metrics, str(path), hashlib.sha256(content).hexdigest(),
                     provenance)
    return result.subset(~np.isin(result.ids, list(exclude_ids)))


def source_groups(dataset):
    """返回来源分组标识
    
    DRIE: 实验2和13共享测量源证据，必须阻断在train/test同侧
    CFD: 每个实验独立
    
    返回：与dataset.ids同长度的分组数组，2和13被标记为同组
    """
    groups = dataset.ids.copy()
    if dataset.name == "real":
        groups[np.isin(dataset.ids, [2, 13])] = 2
    return groups


def make_split(dataset, seed, n_test, n_initial, budget):
    """生成训练/候选池/测试集分割（满足来源阻断约束）
    
    采用拒绝采样确保：
    1. 测试集恰好n_test个样本
    2. 2和13不跨train/test
    3. 初始训练集n_initial个，候选池足够支持budget轮采样
    
    参数：
        seed: 随机种子
        n_test: 测试集大小
        n_initial: 初始训练集大小
        budget: 主动学习预算（采样轮数）
    
    返回：
        {"test": array, "initial": array, "pool": array, "rejection_attempts": int}
    """
    if n_test + n_initial + budget > len(dataset.ids):
        raise ValueError("Budget exceeds available candidate experiments")
    rng = np.random.default_rng(seed)
    groups = source_groups(dataset)
    
    # 拒绝采样：重复随机打乱直到测试集与训练侧无共享来源组
    for attempt in range(10000):
        order = rng.permutation(len(dataset.ids))
        test, remaining = order[:n_test], order[n_test:]
        if not set(groups[test]) & set(groups[remaining]):  # 无交集则成功
            break
    else:
        raise RuntimeError("Could not construct requested source-blocked split")
    
    # 从剩余样本中分配初始训练集和候选池
    initial = np.sort(remaining[:n_initial])
    pool = np.sort(remaining[n_initial:])
    return {"test": np.sort(test), "initial": initial, "pool": pool,
            "rejection_attempts": attempt}


class LabelOracle:
    """标签oracle：按需揭示候选池实验的响应值
    
    主动学习中的标签泄漏防护：
    - 候选池标签不参与模型训练，仅在被选中后揭示
    - 测试集标签仅用于评估，从不揭示给采样策略
    """
    def __init__(self, dataset, initial, pool, test):
        self._dataset = dataset
        self.revealed = set(map(int, initial))  # 已揭示的训练集ID
        self.candidates = set(map(int, pool))   # 候选池ID
        self.test = set(map(int, test))         # 测试集ID（永不揭示）
        if self.revealed & self.test or self.candidates & self.test:
            raise ValueError("Training/pool overlap with held-out test")

    def training(self):
        """返回当前已揭示的训练数据（排序后索引、X、Y）"""
        idx = np.array(sorted(self.revealed), dtype=int)
        return idx, self._dataset.X[idx], self._dataset.Y[idx]

    def reveal(self, index):
        """揭示候选池中指定实验的标签（模拟物理实验执行）
        
        参数：
            index: 数据集中的行索引（非实验ID）
        """
        index = int(index)
        if index not in self.candidates or index in self.test:
            raise ValueError("Only a remaining non-test candidate may be revealed")
        self.candidates.remove(index)
        self.revealed.add(index)
