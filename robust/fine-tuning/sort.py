import re

input_path = "clip_score.txt"  # 替换为你的实际文件路径
output_path = "sorted_clip_scores.txt"

# 读取并解析数据
with open(input_path, 'r') as f:
    lines = f.readlines()

# 提取 checkpoint 数字和原始行内容
data = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    match = re.search(r'checkpoint-(\d+)', line)
    if match:
        checkpoint_num = int(match.group(1))
        data.append((checkpoint_num, line))
    else:
        print(f"无法解析行: {line}")

# 按 checkpoint 数值排序
data.sort(key=lambda x: x[0])

# 写入输出文件
with open(output_path, 'w') as f:
    for _, line in data:
        f.write(line + '\n')

print(f"已按 checkpoint 升序排序并保存至 {output_path}")
