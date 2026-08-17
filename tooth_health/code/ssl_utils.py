"""
自动生成/复用自签名HTTPS证书，给web_app.py用。

为什么需要这个：浏览器调用摄像头(getUserMedia API)要求页面是HTTPS，或者
访问地址是localhost本机——局域网内其他设备(手机/笔记本)通过局域网IP访问
时不满足这两个条件，浏览器会直接拒绝摄像头权限请求，不管代码写得对不对。
自签名证书能让浏览器认可这是"HTTPS"连接(虽然会有"证书不受信任"的警告，
第一次访问需要手动点"继续访问"，这个没法完全避免，因为不是权威CA签发的)。

证书按访问IP生成、缓存在 tooth_health/data/ssl/ 下，IP变了(比如换了个
局域网/换了台机器)会自动重新生成，不用手动跑openssl命令。
"""
import socket
import subprocess
from pathlib import Path


def get_lan_ip() -> str:
    """探测这台机器对外的局域网IP（不实际发送数据，只是借这个UDP连接动作
    让操作系统选一个出口网卡地址）。探测不到就退回127.0.0.1，证书里
    localhost/127.0.0.1这两个SAN始终都会加，本机访问不受影响。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def ensure_self_signed_cert(ssl_dir: Path):
    """返回 (cert_path, key_path)。如果缓存的证书对应的IP跟当前探测到的
    局域网IP不一致（或者证书压根不存在），用openssl重新生成一份，SAN里
    带上当前IP，保证手机/笔记本用这个IP访问时证书是匹配的。"""
    ssl_dir.mkdir(parents=True, exist_ok=True)
    cert_path = ssl_dir / "cert.pem"
    key_path = ssl_dir / "key.pem"
    ip_marker = ssl_dir / ".generated_for_ip"

    current_ip = get_lan_ip()
    cached_ip = ip_marker.read_text().strip() if ip_marker.exists() else None

    if cert_path.exists() and key_path.exists() and cached_ip == current_ip:
        return cert_path, key_path

    print(f"[ssl] 生成自签名证书（局域网IP: {current_ip}），首次访问浏览器会提示"
          "\"证书不受信任\"，这是自签名证书的正常现象，点\"继续访问/高级->继续前往\"就行")
    san = f"subjectAltName=DNS:localhost,IP:127.0.0.1,IP:{current_ip}"
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", str(key_path), "-out", str(cert_path),
             "-days", "3650", "-subj", "/CN=tooth-health-local",
             "-addext", san],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "找不到openssl命令——生成自签名HTTPS证书需要它，Ubuntu下"
            "`sudo apt install openssl`装一下就行，或者用--no_https跳过"
            "（跳过的话只有这台机器自己能用本机摄像头，局域网其他设备的"
            "摄像头会因为浏览器安全策略打不开）")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"生成证书失败: {e.stderr}")

    ip_marker.write_text(current_ip)
    return cert_path, key_path
