"""书稿质量审计：SUMMARY 一致性、章节引用有效性、内容统计

运行：uv run python scripts/audit_book.py
用途：CI 中保障书稿结构完整性（文件齐全、引用有效）
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
issues = []

# 1. SUMMARY.md 中引用的文件必须存在
summary = (SRC / "SUMMARY.md").read_text(encoding="utf-8")
refs = re.findall(r"\(([^)]+\.md)\)", summary)
for ref in refs:
    if not (SRC / ref).exists():
        issues.append(f"SUMMARY 引用缺失文件: {ref}")

# 2. src/ 下的 .md 文件必须都被 SUMMARY 引用
all_md = {str(p.relative_to(SRC)) for p in SRC.rglob("*.md")}
orphan = all_md - set(refs) - {"SUMMARY.md"}
if orphan:
    issues.append(f"未被 SUMMARY 引用的文件: {orphan}")

# 3. 章节交叉引用有效性：正文里"第 N 章"应 ≤ 16
for md in SRC.rglob("*.md"):
    text = md.read_text(encoding="utf-8")
    for n in re.findall(r"第 (\d+) 章", text):
        if int(n) > 16:
            issues.append(f"{md}: 引用了不存在的第 {n} 章")

# 4. 内容统计
stats = {
    "SUMMARY 引用文件": len(refs),
    "python 代码块": sum(len(re.findall(r"```python", p.read_text(encoding="utf-8"))) for p in SRC.rglob("*.md")),
    "mermaid 图": sum(len(re.findall(r"```mermaid", p.read_text(encoding="utf-8"))) for p in SRC.rglob("*.md")),
    "练习题": sum(len(re.findall(r"\d+\. \*\*", p.read_text(encoding="utf-8"))) for p in SRC.rglob("*.md")),
    "性能检查项": sum(len(re.findall(r"- \[ \]", p.read_text(encoding="utf-8"))) for p in SRC.rglob("*.md")),
}

for k, v in stats.items():
    print(f"{k}: {v}")

if issues:
    print("\n发现问题：")
    for i in issues:
        print(f"  - {i}")
    sys.exit(1)
print("\n审计通过")
