# E0 Data Splits — Manifest & Audit

> 生成时间: 2026-06-09T16:14:52.508623
> 生成脚本: `src/data/make_splits_e0.py`
> 项目根目录: /path/to/project

---

## 1. E0 的目的

E0 是项目所有后续实验 (E1–E8) 的**唯一数据划分入口**。所有实验必须使用本目录中的 manifest 文件加载数据, 以保证划分的可复现性和零交集。

## 2. Manifest 文件说明

| 文件 | 用途 | 是否 Paper-Ready |
|------|------|:---:|
| `source_train.txt` | 源域 (WHU aerial) 训练集 | ✅ auxiliary |
| `source_val.txt` | 源域验证集 | ✅ auxiliary |
| `source_test.txt` | 源域测试集 | ✅ auxiliary |
| `target_train_pool_all.txt` | 目标域全部可用训练候选池 | ✅ paper-ready |
| `target_val.txt` | 目标域验证集 | ✅ paper-ready |
| `target_support_*_seed*.txt` | Few-shot support (嵌套) | ✅ paper-ready |
| `target_pilot_test_500.txt` | 试点 500 张测试 | ⚠️ pilot only |
| `target_pilot_test_1000.txt` | 试点 1000 张测试 | ⚠️ pilot only |
| `target_final_test_8402.txt` | 最终论文测试集 | ✅ 8402 full |

## 3. 哪些 split 可用于正式论文

- 所有 target split 均为 paper-ready
- 通过严格审计 (零交集 + 嵌套正确 + 8402 full test)
- 可直接用于 Remote Sensing 期刊投稿

## 4. 哪些 split 只用于 pilot

- `target_pilot_test_500.txt` 和 `target_pilot_test_1000.txt` 仅用于快速调试和模型筛选
- **禁止在 pilot test 上调阈值、调模型、选择 checkpoint**
- 最终论文主结果必须使用 `target_final_test_8402.txt`

## 5. 后续实验如何使用这些 manifest

### E5 (LoRA PEFT) / E6 (GBG-SAM3):
```python
# 读取 support
with open("data/splits/e0_manifest/target_support_10_seed42.txt") as f:
    support_files = [line.strip() for line in f if line.strip()]

# 读取 val
with open("data/splits/e0_manifest/target_val.txt") as f:
    val_files = [line.strip() for line in f if line.strip()]
```

### 注意事项:
- manifest 中每行是**绝对路径**, DataLoader 可直接使用 `cv2.imread(line)`
- 如果现有 Dataset 类不支持 manifest 输入, 需要新增构造参数 `manifest_path`

## 6. 禁止事项

- ❌ 禁止在 final_test 上调阈值
- ❌ 禁止在 final_test 上选择 checkpoint
- ❌ 禁止 support 和 val 之间有交集
- ❌ 禁止 support/val 与 final_test 之间有交集

## 7. 生成命令

```bash
# 当前运行命令:
python src/data/make_splits_e0.py --strict

# 严格 paper-ready 模式:
python src/data/make_splits_e0.py --strict

# 扩展 source + 1000 val:
python src/data/make_splits_e0.py --expand_source --target_val_size 1000
```


---

*此文件由 `make_splits_e0.py` 自动生成, 请勿手动编辑。*
*审计报告详见 `split_audit_report.json`。*
