from PIL import Image
import os

# 输入和输出文件路径
input_path = 'image1.jpg'      # 原始图片路径
output_path = 'resized_image.jpg'   # 缩小后的图片保存路径

# 打开图片
with Image.open(input_path) as img:
    # 缩放为512×512，使用抗锯齿算法保持质量
    resized_img = img.resize((512, 512), Image.LANCZOS)
    # 保存图片
    resized_img.save(output_path)

print(f"图片已成功缩小并保存为: {output_path}")
