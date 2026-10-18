# -*- coding: utf-8 -*-
"""第8课：训练 BPE 分词器（字节级，词表 8000）并导出 C++ 可读的二进制词表。
技巧：把原始字节按 latin-1 解码成字符串喂给 BPE —— 每个 char 恰好是 1 字节，
     merge 出来的 token 再 encode('latin-1') 还原成字节，干净透明。"""
import json, os, struct
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

ROOT = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(ROOT, "cpp")
VOCAB_SIZE = 8000

def rd(p, n=None):
    d = open(os.path.join(CPP, p), "rb").read()
    return d if n is None else d[:n]

def rd_jsonl(path):
    out = []
    for line in open(os.path.join(ROOT, path), encoding="utf-8"):
        out.append(json.loads(line)["对话内容"])
    return "\n\n".join(out).encode("utf-8")

# 【记】训练样本要覆盖模型将来要处理的全部文体
random.seed(42)
parts = [
    rd("hongloumeng.txt"),                    # 古文 2.5MB 全量
    rd("sanguoyanyi.txt") + rd("shuihuzhuan.txt") + rd("xiyouji.txt"),
    rd("zh_modern.txt", 20_000_000),          # 天工现代文 抽20MB
    rd("skypile_cn.txt", 20_000_000),         # 中文网页 抽20MB
    rd("santi1.txt") + rd("santi3.txt"),      # 科幻
    rd_jsonl("日常唠嗑语料_带评分.jsonl"),      # 口语
    rd_jsonl("学术讲评论文问答语料_带评分.jsonl"),
    rd("train_text.txt", 3_000_000),          # 英文 保一点
]
texts = [p.decode("latin-1") for p in parts]   # 1 char = 1 byte
print(f"训练样本合计 {sum(len(t) for t in texts)/1e6:.1f}MB")

tk = Tokenizer(BPE(unk_token=None))
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    initial_alphabet=[chr(i) for i in range(256)],   # 【易错】先把 256 个单字节全塞进词表，任何输入都能编码
    special_tokens=[],
)
tk.train_from_iterator(texts, trainer)

vocab = tk.get_vocab()                 # str -> id（顺序乱，我们要重建）
merges = tk.model.merges               # [(a_str, b_str), ...] 按学习顺序 = 优先级
print(f"词表 {len(vocab)}，merges {len(merges)}")

# ---- 重建 id 体系：0~255 = 单字节，256+k = 第 k 次 merge 的结果 ----
str2bytes = lambda s: s.encode("latin-1")
tok_bytes = [bytes([i]) for i in range(256)]        # id -> 原始字节
id_of = {bytes([i]): i for i in range(256)}
for a, b in merges:
    ab = str2bytes(a) + str2bytes(b)
    assert a in vocab and b in vocab
    assert ab not in id_of, "merge 结果重复？"
    id_of[ab] = len(tok_bytes)
    tok_bytes.append(ab)
assert len(tok_bytes) == len(vocab), (len(tok_bytes), len(vocab))

# ---- 导出二进制：magic | vocab_size | (len,bytes)... | merges_count | (a_id,b_id)... ----
merges_id = [(id_of[str2bytes(a)], id_of[str2bytes(b)]) for a, b in merges]
out = os.path.join(CPP, "bpe_model.bin")
with open(out, "wb") as f:
    f.write(b"BPE1")
    f.write(struct.pack("<i", len(tok_bytes)))
    for tb in tok_bytes:
        f.write(struct.pack("<H", len(tb)))
        f.write(tb)
    f.write(struct.pack("<i", len(merges_id)))
    for a, b in merges_id:
        f.write(struct.pack("<ii", a, b))
print(f"导出 {out}  ({os.path.getsize(out)/1e6:.2f}MB)")

# ---- 抽查：常见中文被切成什么样 ----
def show(s):
    enc = tk.encode(s.encode("utf-8").decode("latin-1"))
    ids = enc.ids
    print(f"  {s!r}: {len(s.encode('utf-8'))} 字节 -> {len(ids)} token")
    pieces = [tok_bytes[i] for i in ids]
    print("   切分:", " | ".join(p.decode("utf-8", errors="replace") for p in pieces))

print("切分抽查：")
for s in ["你好，世界", "林黛玉进贾府", "人工智能正在改变世界", "今天天气怎么样？", "to be or not to be"]:
    show(s)
