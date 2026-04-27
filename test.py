import pickle
import torch
import sys
from ocr import ocr
import numpy as np
from PIL import Image

# 1. 加载字符集
with open('recognize\\alphabet.pkl', 'rb') as f:
    alphabet = pickle.load(f)
print(f"字符集大小: {len(alphabet)}")
print(f"前20个字符: {alphabet[:20]}")
print(f"中文示例: {alphabet[100:120]}")

# 2. 测试一张图片并查看原始输出
filename = sys.argv[1] if len(sys.argv) > 1 else "test_images/t1.png"
img = np.array(Image.open(filename).convert('RGB'))

print("\n正在识别...")
result, framed = ocr(img)

print("\n详细结果分析:")
print("="*60)
for idx, key in enumerate(result):
    text = result[key][1]
    print(f"\n区域 {idx+1}:")
    print(f"  输出文本: {text}")
    print(f"  文本长度: {len(text)}")
    
    # 检查每个字符是否在字符集中
    if len(text) > 0:
        print("  字符分析:")
        for i, ch in enumerate(text[:20]):  # 只显示前20个
            if ch in alphabet:
                pos = alphabet.index(ch)
                print(f"    {i}: '{ch}' 在字符集中 (索引 {pos})")
            else:
                print(f"    {i}: '{ch}' 不在字符集中!")

print("="*60)