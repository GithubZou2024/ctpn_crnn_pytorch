from flask import Flask, request, jsonify, render_template_string
import base64
import numpy as np
from PIL import Image
import cv2
import webbrowser
import threading
from ocr import ocr

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR 识别系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f9f9f9;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: #f0f0ff;
        }
        .result-area {
            display: flex;
            gap: 30px;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        .image-box, .text-box {
            flex: 1;
            min-width: 300px;
        }
        .image-box {
            background: #f5f5f5;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        .text-box {
            background: #f5f5f5;
            border-radius: 15px;
            padding: 20px;
        }
        .result-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .recognized-text {
            background: white;
            padding: 20px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 14px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow-y: auto;
            margin: 15px 0;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background: #5a67d8;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .btn-clear {
            background: #ccc;
        }
        .btn-clear:hover {
            background: #bbb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 OCR 文字识别系统</h1>
        <div class="subtitle">基于 PyTorch CTPN + CRNN | 支持中英文识别</div>
        
        <div class="upload-area" id="uploadArea">
            <div>📸 点击或拖拽上传图片</div>
            <div style="font-size: 12px; color: #999; margin-top: 10px;">支持 JPG、PNG 格式</div>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在识别中，请稍候...</p>
        </div>
        
        <div class="result-area" id="resultArea" style="display: none;">
            <div class="image-box">
                <h3>识别结果图片</h3>
                <img id="resultImage" class="result-image" alt="识别结果">
                <div class="btn-group">
                    <button onclick="downloadImage()">💾 下载图片</button>
                </div>
            </div>
            <div class="text-box">
                <h3>识别文字</h3>
                <div id="recognizedText" class="recognized-text"></div>
                <div class="btn-group">
                    <button onclick="copyText()">📋 复制文字</button>
                    <button onclick="downloadText()">💾 下载文字</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const resultArea = document.getElementById('resultArea');
        const resultImage = document.getElementById('resultImage');
        const recognizedText = document.getElementById('recognizedText');
        
        let currentImageData = null;
        
        uploadArea.onclick = () => fileInput.click();
        
        uploadArea.ondragover = (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#667eea';
        };
        
        uploadArea.ondragleave = () => {
            uploadArea.style.borderColor = '#ccc';
        };
        
        uploadArea.ondrop = async (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#ccc';
            const file = e.dataTransfer.files[0];
            if (file) await uploadImage(file);
        };
        
        fileInput.onchange = async (e) => {
            if (e.target.files[0]) await uploadImage(e.target.files[0]);
        };
        
        async function uploadImage(file) {
            const formData = new FormData();
            formData.append('image', file);
            
            loading.style.display = 'block';
            resultArea.style.display = 'none';
            
            try {
                const response = await fetch('/ocr', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultImage.src = 'data:image/jpeg;base64,' + data.image;
                    recognizedText.textContent = data.full_text;
                    currentImageData = data.image;
                    resultArea.style.display = 'flex';
                } else {
                    alert('识别失败: ' + data.error);
                }
            } catch (error) {
                alert('请求失败: ' + error);
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function copyText() {
            const text = recognizedText.textContent;
            navigator.clipboard.writeText(text);
            alert('已复制到剪贴板');
        }
        
        function downloadImage() {
            const link = document.createElement('a');
            link.download = 'ocr_result.jpg';
            link.href = resultImage.src;
            link.click();
        }
        
        function downloadText() {
            const text = recognizedText.textContent;
            const blob = new Blob([text], {type: 'text/plain'});
            const link = document.createElement('a');
            link.download = 'ocr_result.txt';
            link.href = URL.createObjectURL(blob);
            link.click();
            URL.revokeObjectURL(link.href);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ocr', methods=['POST'])
def ocr_process():
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)
        
        # 调用您的 OCR 函数
        result, image_framed = ocr(image_np)
        
        # 编码图片为 base64
        _, buffer = cv2.imencode('.jpg', image_framed)
        image_base64 = base64.b64encode(buffer).decode()
        
        # 提取所有文字
        texts = []
        for key in result:
            text = result[key][1]
            texts.append(text)
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'full_text': '\n'.join(texts)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # 自动打开浏览器
    def open_browser():
        webbrowser.open('http://127.0.0.1:5000')
    
    print("\n" + "="*50)
    print("🚀 OCR Web 服务启动中...")
    print("="*50)
    print("📱 请在浏览器中打开: http://127.0.0.1:5000")
    print("🔄 服务运行中，按 Ctrl+C 停止")
    print("="*50 + "\n")
    
    # 延迟1秒打开浏览器
    threading.Timer(1, open_browser).start()
    
    # 启动服务器
    app.run(debug=False, host='127.0.0.1', port=5000)