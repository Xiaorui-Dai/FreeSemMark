import matplotlib.pyplot as plt
import re
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'


input_file = 'clip_score.txt'

checkpoints = []
scores = []

with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'checkpoint-(\d+):\s*([\d.]+)', line)
        if match:
            checkpoint = int(match.group(1))
            score = float(match.group(2))
            checkpoints.append(checkpoint)
            scores.append(score)
        else:
            print(f"格式错误，跳过该行: {line}")

if not checkpoints:
    print("未提取到有效数据，请检查文件格式")
    exit()

# 排序
checkpoints, scores = zip(*sorted(zip(checkpoints, scores)))

# 绘图
plt.figure(figsize=(5.5, 3.8))
plt.plot(checkpoints, scores, marker='o', linestyle='-', color='#B73E3E')
plt.xlabel('Steps',fontsize=12)
plt.ylabel(r'$S_{\text{forward}}$',fontsize=12)

# ✅ 设置背景网格为虚线
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# ✅ 设置 y 轴从 0 开始
plt.ylim(0,0.30)
plt.xlim(0,10500)

plt.tight_layout()
plt.savefig("clip_score_plot.png", dpi=600,bbox_inches='tight')
plt.show()
