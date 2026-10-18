# -*- coding: utf-8 -*-
"""训练样本盘点：四大名著 + FineWeb-Edu"""
import os, json
from collections import Counter

CPP = r"C:\Users\yangz\Documents\Kimi\Workspaces\4小时学习\cpp"
FINEWEB = r"C:\Users\yangz\Documents\fineweb-edu-sample-10BT"

# ---------- 1. 四大名著 ----------
books = [("红楼梦", "hongloumeng.txt"), ("三国演义", "sanguoyanyi.txt"),
         ("水浒传", "shuihuzhuan.txt"), ("西游记", "xiyouji.txt")]
all_chars = Counter()
report = []
total_hanzi = 0
for name, fn in books:
    raw = open(os.path.join(CPP, fn), "rb").read()
    text = raw.decode("utf-8", errors="ignore")
    hanzi = [c for c in text if '\u4e00' <= c <= '\u9fff']
    cnt = Counter(hanzi)
    all_chars.update(cnt)
    total_hanzi += len(hanzi)
    top = cnt.most_common(8)
    report.append({
        "书名": name, "文件": fn, "字节数": len(raw), "汉字数": len(hanzi),
        "不重复汉字": len(cnt), "高频字": top,
    })

print("=== 四大名著 ===")
for r in report:
    print(f"{r['书名']}: {r['字节数']/1e6:.2f}MB, 汉字 {r['汉字数']:,}, "
          f"不重复 {r['不重复汉字']:,} 种")
    print("   高频:", " ".join(f"{c}×{n}" for c, n in r["高频字"][:6]))
print(f"合计汉字 {total_hanzi:,}，四本书合并不重复汉字 {len(all_chars):,} 种")
print("合并高频:", " ".join(f"{c}×{n}" for c, n in all_chars.most_common(10)))

# ---------- 2. FineWeb-Edu ----------
print("\n=== FineWeb-Edu ===")
import duckdb
con = duckdb.connect()
files = sorted(f for f in os.listdir(FINEWEB) if f.endswith(".parquet"))
print(f"分片数: {len(files)}")
total_rows, total_bytes = 0, 0
for f in files:
    p = os.path.join(FINEWEB, f)
    sz = os.path.getsize(p)
    total_bytes += sz
    n = con.execute(f"select count(*) from '{p}'").fetchone()[0]
    total_rows += n
print(f"总大小 {total_bytes/1e9:.1f} GB, 总文档 {total_rows:,} 篇")

# schema & score 分布（用第一个分片）
p0 = os.path.join(FINEWEB, files[0])
cols = [d[0] for d in con.execute(f"describe select * from '{p0}'").fetchall()]
print("列名:", cols)
if "score" in cols:
    df = con.execute(f"select score from '{p0}' where random() < 0.01").df()
    import pandas as pd
    bins = pd.cut(df["score"], bins=[0,1,2,3,4,5], include_lowest=True)
    print("score(教育性评分)分布 (分片0, 1%抽样):")
    print(bins.value_counts().sort_index().to_string())
