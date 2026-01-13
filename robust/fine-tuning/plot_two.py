import matplotlib.pyplot as plt
import re
from matplotlib import rcParams

# 字体设置
rcParams['font.family'] = 'Times New Roman'

def read_scores(file_path):
    checkpoints = []
    scores = []
    with open(file_path, 'r') as f:
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
        print(f"{file_path} 中未提取到有效数据，请检查文件格式")
        return None, None

    # 排序
    checkpoints, scores = zip(*sorted(zip(checkpoints, scores)))
    return checkpoints, scores

# 读取两组数据
ckpt_full, score_full = read_scores('clip_score_full.txt')
ckpt_lora, score_lora = read_scores('clip_score_lora.txt')

# 绘图
plt.figure(figsize=(5.5, 3.8))

# full-parameter
plt.plot(ckpt_full, score_full, marker='o', linestyle='-', color='#B73E3E', label='Full-Parameter')

# FoRA
plt.plot(ckpt_lora, score_lora, marker='s', linestyle='-', color='#227C9D', label='LoRA')

# 轴标签
plt.xlabel('Steps', fontsize=12)
plt.ylabel(r'$S_{\text{forward}}$', fontsize=12)

# 网格
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 坐标范围
plt.ylim(0, 0.30)
plt.xlim(0, 10500)

# 图例
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("clip_score_plot.png", dpi=600, bbox_inches='tight')
plt.show()
