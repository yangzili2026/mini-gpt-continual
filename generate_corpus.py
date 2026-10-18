# -*- coding: utf-8 -*-
"""生成带评分的日常口语唠嗑语料（用于大模型训练）"""
import json, random, os

random.seed(42)

# 话题库：每个话题有开场、推进、回应、收尾四种话术槽位
TOPICS = {
    "吃饭": {
        "openers": ["哎你吃了没？", "下班吃啥去？", "今天午饭你搁哪儿解决的？"],
        "advancers": ["我刚点了个外卖，等了四十分钟，饿死我了。", "楼下那家面馆最近换老板了，味儿都不一样了。", "我妈老说我点外卖不健康，可她做的饭我也吃够了呀。"],
        "responders": ["哈哈我也是，天天纠结吃啥。", "那你晚上自己做啊？", "我家楼下那家挺好吃，推荐你试试。"],
        "closers": ["行，改天一起去尝尝。", "得，先忙去了啊。", "回头聊，我先扒口饭。"],
        "formality": 0.9
    },
    "天气": {
        "openers": ["今天这天儿也太冷了。", "明儿是不是要下雨啊？", "这鬼天气，说变就变。"],
        "advancers": ["我早上出门穿少了，冻得直哆嗦。", "衣服晾了三天都没干，愁人。", "这天儿一冷，我就想吃火锅。"],
        "responders": ["那你多穿点啊。", "我看天气预报说明儿降温。", "降温好啊，至少不用开空调了，省电。"],
        "closers": ["得，出门记得带伞。", "行，注意保暖啊。", "回见。"],
        "formality": 0.85
    },
    "上班": {
        "openers": ["今儿上班没迟到吧？", "最近忙不忙？", "周一又要开那个破会，烦死。"],
        "advancers": ["我们领导又改需求了，上周五做的全白搭。", "加班到九点，一分钱加班费没有。", "新来的那个同事人还挺好，天天给我带咖啡。"],
        "responders": ["我们这儿也是，领导一张嘴，底下跑断腿。", "那你跟领导说说呗。", "不错了，我这天天挨骂。"],
        "closers": ["行了行了，都不容易。", "熬着吧，熬到退休。", "先干活去了，回聊。"],
        "formality": 0.88
    },
    "孩子": {
        "openers": ["你家孩子期中考试咋样？", "娃最近乖不乖？", "小孩上幼儿园适应不？"],
        "advancers": ["数学考了八十多，语文差点，愁死我了。", "天天追着喂饭，一顿饭能吃一小时。", "给报了个画画班，他倒挺乐意去。"],
        "responders": ["不错了，我家那个及格都费劲。", "小孩嘛，大了就好了。", "别报太多班，孩子累。"],
        "closers": ["得，回头交流交流育儿经。", "行，接孩子去了啊。", "慢慢来吧。"],
        "formality": 0.82
    },
    "网购": {
        "openers": ["你双十一买啥了？", "最近有啥好东西推荐不？", "我昨天又剁手了。"],
        "advancers": ["凑单凑到凌晨两点，省了三十块，多花三百。", "买了个空气炸锅，用了两次就吃灰了。", "直播间说得太好了，没忍住就下单了。"],
        "responders": ["哈哈一样一样，购物车永远清不空。", "空气炸锅我家也闲置着。", "退货呗，七天无理由。"],
        "closers": ["行，链接发我一个。", "我先去收快递了。", "下回有便宜货喊我。"],
        "formality": 0.9
    },
    "减肥": {
        "openers": ["你最近瘦了啊？", "我下决心减肥了，真的。", "晚上别叫我吃饭啊，我减肥。"],
        "advancers": ["坚持了三天，今晚没忍住吃了顿烧烤。", "办了张健身卡，去了两次就不去了。", "每天走一万步，秤是一点没动。"],
        "responders": ["减啥肥，你又不胖。", "烧烤配啤酒，快乐翻倍。", "管住嘴比啥都强。"],
        "closers": ["哈哈行，互相监督。", "减肥是终生事业，不急。", "得了，梦里啥都有。"],
        "formality": 0.9
    },
    "房价": {
        "openers": ["听说你们小区又跌了？", "现在这房价，到底该不该买啊？", "租房可真是个无底洞。"],
        "advancers": ["我同事去年买的，今年首付都跌没了。", "看了半年房，越看越不敢下手。", "房东又要涨价，真想咬咬牙买了。"],
        "responders": ["再等等呗，反正不急。", "刚需就买，涨跌跟你关系不大。", "租房也挺好，想去哪儿去哪儿。"],
        "closers": ["再观望观望吧。", "行，到时候互相通气。", "唉，先活着吧。"],
        "formality": 0.75
    },
    "追剧": {
        "openers": ["你看那个新剧了没？", "昨晚一口气追到三点。", "有啥剧推荐不？剧荒了。"],
        "advancers": ["男女主终于在一起了，我哭得稀里哗啦。", "那个反派演得真好，恨得我牙痒痒。", "结局烂尾了，白追四十集。"],
        "responders": ["我也在看！更新太慢了。", "倍速看的，剧情太拖。", "看解说版得了，省时间。"],
        "closers": ["行，我今晚也追追。", "有好剧记得安利我。", "别给我剧透啊！"],
        "formality": 0.92
    },
    "老家": {
        "openers": ["你啥时候回老家啊？", "今年国庆回去没？", "我妈又催我回家相亲。"],
        "advancers": ["村里路修好了，回家方便多了。", "我爸养了十几只鸡，天天视频给我看他的鸡。", "一回去就被问工资问对象，头大。"],
        "responders": ["哈哈全国统一的爸妈。", "那挺好啊，等我有空也去你老家玩玩。", "相亲就去呗，万一呢。"],
        "closers": ["行，回去替我问叔叔阿姨好。", "得了，先躲躲清净。", "回头一起回呗。"],
        "formality": 0.85
    },
    "游戏": {
        "openers": ["上号？今晚开黑。", "你那个游戏打到啥段位了？", "昨晚连败十把，气死我了。"],
        "advancers": ["队友太菜了，带不动。", "新赛季皮肤挺好看，想氪金了。", "我媳妇不让玩，只能等她睡了偷摸打两把。"],
        "responders": ["哈哈那你小心跪键盘。", "什么段位，别是青铜吧。", "来个辅助位，我射手贼六。"],
        "closers": ["九点准时上线。", "行，晚上见。", "先不聊了，我这把打团呢。"],
        "formality": 0.95
    },
}

# 语气词 / 口语填充成分（提升口语化度）
FILLERS = ["唉", "哎呀", "嗐", "啧", "诶", "哈哈", "嘿嘿", "天哪", "说真的", "说实话", "你别说"]

def build_dialogue(topic_key):
    t = TOPICS[topic_key]
    turns = [("A", random.choice(t["openers"]))]
    # 打乱槽位池，避免同一对话内句子重复
    responders = random.sample(t["responders"], len(t["responders"]))
    advancers = random.sample(t["advancers"], len(t["advancers"]))
    k = random.randint(2, 3)
    for i in range(k):
        turns.append(("B", responders[i % len(responders)]))
        turns.append(("A", advancers[i % len(advancers)]))
    turns.append(("B", random.choice(t["closers"])))
    # 随机注入语气词
    if random.random() < 0.6:
        i = random.randrange(len(turns))
        speaker, text = turns[i]
        turns[i] = (speaker, random.choice(FILLERS) + "，" + text)
    return turns

def score_dialogue(turns, topic_key):
    t = TOPICS[topic_key]
    text = "".join(x[1] for x in turns)
    # 自然度：口语特征密度（语气词、短句、生活化词汇）
    colloquial_hits = sum(1 for w in ["了","呢","啊","呗","呀","嘛","搁","啥","咋","唠","哎"] if w in text)
    naturalness = min(5.0, round(3.2 + 0.25 * colloquial_hits + random.uniform(-0.2, 0.2), 1))
    # 流畅度：平均句长（过短或过长都扣分）
    import statistics
    lens = [len(x[1]) for x in turns]
    mean_len = statistics.mean(lens)
    fluency = min(5.0, round(5.0 - abs(mean_len - 14) * 0.08 + random.uniform(-0.15, 0.15), 1))
    # 口语化程度
    colloquial = min(5.0, round(2.5 + 0.5 * colloquial_hits * t["formality"] + random.uniform(-0.2, 0.2), 1))
    # 信息量：话题推进轮数
    informativeness = min(5.0, round(2.0 + 0.6 * (len(turns) - 2) + random.uniform(-0.2, 0.2), 1))
    # 综合质量分（加权）
    overall = round(naturalness * 0.35 + fluency * 0.25 + colloquial * 0.25 + informativeness * 0.15, 1)
    return {
        "自然度(1-5)": max(1.0, naturalness),
        "流畅度(1-5)": max(1.0, fluency),
        "口语化程度(1-5)": max(1.0, colloquial),
        "信息量(1-5)": max(1.0, informativeness),
        "综合质量分(1-5)": max(1.0, min(5.0, overall)),
    }

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    records = []
    n = 500
    topic_keys = list(TOPICS.keys())
    for i in range(n):
        tk = random.choice(topic_keys)
        turns = build_dialogue(tk)
        scores = score_dialogue(turns, tk)
        records.append({
            "id": f"chat-{i+1:04d}",
            "场景": "日常唠嗑",
            "话题": tk,
            "对话轮数": len(turns),
            "对话内容": "\n".join(f"{s}：{t}" for s, t in turns),
            **scores,
        })

    jsonl_path = os.path.join(out_dir, "日常唠嗑语料_带评分.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Excel 版
    import openpyxl
    from openpyxl.styles import Font, PatternFill
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "唠嗑语料"
    headers = list(records[0].keys())
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="D9E1F2")
    for r in records:
        ws.append([r[h] for h in headers])
    ws.column_dimensions["F"].width = 70
    ws.freeze_panes = "A2"
    xlsx_path = os.path.join(out_dir, "日常唠嗑语料_带评分.xlsx")
    wb.save(xlsx_path)

    print(f"生成 {len(records)} 条语料")
    print(jsonl_path)
    print(xlsx_path)

if __name__ == "__main__":
    main()
