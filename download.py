import requests
import os
from concurrent.futures import ThreadPoolExecutor

def download_video(number):
    base_url = "https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects/user-cloud-storage/kibt_creation"
    filename = f"{number}.mp4"
    url = f"{base_url}/{filename}"
    
    try:
        # 这里需要添加你的认证信息
        headers = {
            # 'Cookie': '你的认证cookie',
            # 'Authorization': '你的认证token'
        }
        
        print(f"开始下载 {filename}")
        response = requests.get(url, headers=headers, stream=True)
        
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"成功下载 {filename}")
        else:
            print(f"下载失败 {filename}: 状态码 {response.status_code}")
            
    except Exception as e:
        print(f"下载 {filename} 时发生错误: {str(e)}")

def main():
    # 创建一个线程池来并行下载
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 下载4.mp4到15.mp4
        executor.map(download_video, range(4, 16))

if __name__ == "__main__":
    main()