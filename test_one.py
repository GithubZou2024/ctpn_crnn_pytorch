import os
import sys
import numpy as np
from PIL import Image
import cv2
from ocr import ocr

def single_pic_proc(image_file):
    """处理单张图片"""
    print(f"📷 正在处理: {image_file}")
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result, image_framed

if __name__ == '__main__':
    # 方式1：命令行参数输入
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        # 方式2：手动输入路径
        print("="*50)
        print("OCR 图片识别")
        print("="*50)
        filename = input("请输入图片路径: ").strip()
        # 去掉可能的引号
        filename = filename.strip('"').strip("'")
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"❌ 文件不存在: {filename}")
        print("\n使用示例:")
        print("  python test_one_input.py test_image/t1.png")
        print("  python test_one_input.py C:/Users/xxx/pic.jpg")
        sys.exit(1)
    
    # 检查文件格式
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        print(f"❌ 不支持的文件格式，请使用 jpg 或 png 图片")
        sys.exit(1)
    
    # 执行OCR
    try:
        result, image_framed = single_pic_proc(filename)
        result_dir = "./test_result"
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_image = os.path.join(result_dir, f"{base_name}_result.jpg")
        output_txt = os.path.join(result_dir, f"{base_name}.txt")
        
        # 保存带框的图片
        cv2.imwrite(output_image, image_framed)
        
        # 保存识别文字并打印
        print("\n" + "="*50)
        print("📝 识别结果:")
        print("="*50)
        
        all_text = []
        with open(output_txt, 'w', encoding='utf-8') as f:
            for idx, key in enumerate(result, 1):
                text = result[key][1]
                print(f"{idx}. {text}")
                f.write(text + '\n')
                all_text.append(text)
        
        print("="*50)
        print(f"\n✅ 完成!")
        print(f"📁 带框图片: {output_image}")
        print(f"📄 识别文字: {output_txt}")
        
        # 询问是否显示图片
        show = input("\n是否打开查看图片？(y/n): ").lower()
        if show == 'y':
            try:
                cv2.imshow('OCR Result - 按任意键关闭', image_framed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("无法显示图片窗口")
        
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        import traceback
        traceback.print_exc()