from modelscope import snapshot_download
import os
import sys

def main():
    try:
        print("开始下载SenseVoice模型...")
        print("这可能需要几分钟时间，请耐心等待...")

        # Download model
        model_dir = snapshot_download('iic/SenseVoice', cache_dir='./models')

        print(f"✅ 模型下载完成！")
        print(f"模型路径: {model_dir}")

        # Check model size
        import shutil
        model_size = shutil.disk_usage(model_dir).used / (1024**3)  # Convert to GB
        print(f"模型大小: {model_size:.2f} GB")

        return 0

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
