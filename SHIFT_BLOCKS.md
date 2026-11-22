# 双班段离线合算

两班的原测分别归档。值班员从每个站带走一份班段件，中心只需这两份件以及已经登记的公共模型。原有模型、观测修订、单串冻结及报告接口见 README.md。

## 交件和领取

```python
from observations.blocks import issue_block
from analysis_tasks.blocks import sign_blocks, read_blocks

left = issue_block(day_station, run_id, start, model_ref, day_revision, day_count)
right = issue_block(night_station, run_id, start + day_count, model_ref, night_revision, night_count)
# 两站关闭后，只使用 center 和可携带的两件。
report = sign_blocks(center, report_id, left, right, before_commit=None)
recovered = read_blocks(center, report_id)
```

`source` 和 `center` 均用 `archive.connect` 打开。`run_id` 和 `report_id` 为非空字符串。`start` 是调用方给定的非负整数运行序位，`prefix` 为指定 `SequenceRevision` 的合法正整数前缀长度，布尔值无效。各前缀内沿用 V1 的记录排序，跨件邻接只按运行序位判定。前班的起拍可以大于零，平衡截取仍从前班首拍开始。用 `register_model` 在各站和中心登记同一个模型修订及内容。站端 `issue_block(source, run_id, start, model_ref, sequence_revision, prefix)` 只使用本站数据，按给定引用取数，未知引用沿用 V1 的查找异常。

班段件为能经标准 JSON 序列化及读回的字典。件头有 `block_id`、`run_id`、`start`、`length`、`model_ref`、`sequence_ref`、`snapshot_id`。`length` 是前缀拍数，`model_ref` 是 `[修订号, 内容指纹]`，`sequence_ref` 是 `[序列号, 版本号]`。`snapshot_id` 引用确切前缀。`block_id` 是绑定全部件头出处与分析正文的稳定内容身份，同一冻结前缀及交件请求重复发行保持一致。分析正文的字段、表示和版本标记由实现定义。只交班段级分析证据，逐拍观测及逐拍似然留在站库。同一张件可直接在任一班位使用。本站前缀自身真零时可沿用 `ZeroEvidenceError` 拒发。

`sign_blocks(center, report_id, left, right, *, before_commit=None)` 核对件结构、证据与件头的内容绑定、共同运行号及确切模型引用。模型须能在中心解析，左件 `start + length` 须等于右件 `start`。坏件、错配、逆序、间断、重叠和非法参数报 `ValueError`。中心无需打开任何站库。两件共同无支持时抛 `duration_inference.evidence.ZeroEvidenceError`。使用已发布的报告号和同一有序双件请求时返回原定稿及回执，异请求抛 `analysis_tasks.catalog.TaskIdConflict`。

## 统计及归档

沿用 V1 平衡截取和真实更新口径。左首按平衡剩余寿命取状态，右首从左末经过一次模型转移。占用按发射拍归班，更新按新段的到达拍归班。同状态新段照计，整串首拍及整串末拍以后不计更新。两班所有观测共同决定路径的条件质量。

`sign_blocks` 与 `read_blocks` 返回独立字典。`report_id` 是申请号，`run_id` 是共同运行，`model_ref` 是实际模型引用，`blocks` 是有序的两份件头。`states` 按登记顺序给出状态。`dimensions`、`mean`、`covariance` 沿用 V1 的形状及精度，每班依次列各状态的 `occupancy` 与该班的 `renewals`，班号先 0 后 1。`evidence` 沿用 V1 的 `mantissa`、`exponent`、`log_evidence`、`probability`、`probability_representable`，表示两班共用的观测概率。极小正证据仍可签发。`receipt` 包含此报告的 `report_id` 及稳定的 `receipt_id`。

正文和回执同次提交。提交钩子沿用 README.md 的调用时点与限制，钩子抛错用 `run_reports.publication.PublicationError` 保留原异常原因，本次报告与回执均无新增，原申请可重办。`read_blocks(center, report_id)` 从归档读定稿，更正站端记录或重开中心库后仍按原号读存值。合法号查无定稿抛 `run_reports.reader.ReportNotPublished`。原 V1 接口、归档结果与回归照常。

## 联合核账

`block_examples.py` 提供模型、班段符号和单串 V1 对照。直接运行它可核对以下业务值，其中 `|` 为分班点。

| 样例 | 应得结果 |
| --- | --- |
| `xx|x`，恒定两拍一段 | 证据为 1，均值为 `[2, 1/2, 1, 1/2]`，跨班更新协方差为 `−1/4`。 |
| `ab|a`，确定性交替 | 证据为 `1/2`，后班更新恰为 1。 |
| `ba|a`，同一交替模型 | 左班旧报的证据及统计与 `ab` 相同，但拼接后共同支持为零，拒签。 |
| 两班各 20 拍 `x`，单拍概率 `10⁻¹⁰` | 证据为 `10⁻⁴⁰⁰`，`probability` 为 `None`，照常签发。 |

交付时在现有 `.pointwise-runtime.json` 的 `verify` 入口接通原回归与班段联合核验，失败须以非零退出码结束。`demo` 展示各站独立交件、关站后合算以及重开中心库后的原号领取。样例的整串对照只用来核账。
