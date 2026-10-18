# -*- coding: utf-8 -*-
"""从 FineWeb-Edu 切一小片高质量英文文本当训练语料"""
import duckdb
import pathlib

SRC = r"C:\Users\yangz\Documents\fineweb-edu-sample-10BT\000_00000.parquet"
OUT = pathlib.Path(r"C:\Users\yangz\Documents\Kimi\Workspaces\4小时学习\train_text.txt")

TARGET_CHARS = 12_000_000  # 约 1200 万字符

con = duckdb.connect()
rows = con.execute(f"""
    SELECT text, score, token_count
    FROM '{SRC}'
    WHERE language = 'en' AND score >= 4.0 AND token_count BETWEEN 50 AND 800
    USING SAMPLE 60000 ROWS
""").fetchall()

docs, total = [], 0
for text, score, tc in rows:
    text = text.strip()
    if total + len(text) > TARGET_CHARS:
        continue
    docs.append(text)
    total += len(text)

OUT.write_text("\n\n".join(docs), encoding="utf-8")
print(f"文档数: {len(docs)}")
print(f"总字符: {total:,}")
print(f"文件大小: {OUT.stat().st_size/1e6:.1f} MB")
print("--- 第一篇的开头 300 字 ---")
print(docs[0][:300])
