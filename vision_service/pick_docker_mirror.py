#!/usr/bin/env python3
"""挑最快的 Docker Hub 加速站：各拉同一个小镜像的一层 3 秒，按速度排。

    python vision_service/pick_docker_mirror.py            → 打印各站速度，最后一行是选中的（缓存一天）
    python vision_service/pick_docker_mirror.py --force    → 不看缓存重测
    python vision_service/pick_docker_mirror.py --quiet    → 只打印选中的那个（给脚本用）

为什么不改 /etc/docker/daemon.json 的 registry-mirrors：改了要重启 docker，label_service 那些
容器会跟着断一下；而且国内加速站时好时坏，写死一个隔天就可能不通。这里每次拉大镜像前测一遍，
用「<加速站>/vllm/vllm-openai:latest」这种带前缀的名字去 pull，拉完 docker tag 回原名，daemon 不用动。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

CACHE = os.environ.get("DOCKER_MIRROR_CACHE") or os.path.expanduser("~/.cache/imu_train_docker_mirror")
TTL = 86400
PROBE_REPO = "library/alpine"
PROBE_TAG = "3.19"
# 候选：加速站（拉 docker hub 的镜像时把它当前缀），最后是官方
CANDIDATES = [
    ("官方 docker.io", "registry-1.docker.io", ""),
    ("DaoCloud", "docker.m.daocloud.io", "docker.m.daocloud.io"),
    ("1ms.run", "docker.1ms.run", "docker.1ms.run"),
    ("南京大学", "docker.nju.edu.cn", "docker.nju.edu.cn"),
    ("1Panel", "docker.1panel.live", "docker.1panel.live"),
    ("轩辕", "docker.xuanyuan.me", "docker.xuanyuan.me"),
    ("hub.rat.dev", "hub.rat.dev", "hub.rat.dev"),
    ("dockerproxy", "dockerproxy.net", "dockerproxy.net"),
]
UA = {"User-Agent": "imu-train-mirror-probe"}


def _get(url: str, headers: dict, timeout: float):
    req = urllib.request.Request(url, headers={**UA, **headers})
    return urllib.request.urlopen(req, timeout=timeout)


def _token(host: str, timeout: float = 5) -> str | None:
    """走一遍 /v2/ 的 401 → WWW-Authenticate → 拿匿名 token（各站要不要 token 不一样）。"""
    try:
        _get(f"https://{host}/v2/", {}, timeout)
        return None                      # 不要 token
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        auth = e.headers.get("WWW-Authenticate", "")
    m = re.search(r'realm="([^"]+)"', auth)
    if not m:
        return None
    realm = m.group(1)
    svc = re.search(r'service="([^"]+)"', auth)
    q = f"?scope=repository:{PROBE_REPO}:pull" + (f"&service={svc.group(1)}" if svc else "")
    with _get(realm + q, {}, timeout) as r:
        return json.loads(r.read().decode()).get("token") or json.loads(r.read().decode()).get("access_token")


def _layer_digest(host: str, tok: str | None, timeout: float = 6) -> str | None:
    h = {"Accept": "application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json, "
                   "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    with _get(f"https://{host}/v2/{PROBE_REPO}/manifests/{PROBE_TAG}", h, timeout) as r:
        m = json.loads(r.read().decode())
    if "manifests" in m:              # 多架构索引 → 取 amd64 那份
        d = next((x["digest"] for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"), None)
        if not d:
            return None
        with _get(f"https://{host}/v2/{PROBE_REPO}/manifests/{d}", h, timeout) as r:
            m = json.loads(r.read().decode())
    layers = m.get("layers") or []
    return layers[0]["digest"] if layers else None


def measure(host: str, seconds: float = 3.0) -> float:
    """下载那一层最多 seconds 秒，返回 字节/秒；不通返回 0。"""
    try:
        tok = _token(host)
        digest = _layer_digest(host, tok)
        if not digest:
            return 0.0
        h = {"Authorization": f"Bearer {tok}"} if tok else {}
        t0 = time.monotonic()
        got = 0
        with _get(f"https://{host}/v2/{PROBE_REPO}/blobs/{digest}", h, 8) as r:
            while time.monotonic() - t0 < seconds:
                chunk = r.read(65536)
                if not chunk:
                    break
                got += len(chunk)
        dt = max(1e-3, time.monotonic() - t0)
        return got / dt
    except Exception:  # noqa: BLE001 不通就是不通
        return 0.0


def pick(force: bool = False, quiet: bool = False) -> str:
    if not force and os.path.isfile(CACHE) and time.time() - os.path.getmtime(CACHE) < TTL:
        with open(CACHE) as f:
            return f.read().strip()
    best, best_bps = "", 0.0
    if not quiet:
        print("测各 Docker Hub 加速站速度（各 3 秒）...", flush=True)
    for name, host, prefix in CANDIDATES:
        bps = measure(host)
        if not quiet:
            print(f"  {name:<14} {bps / 1048576:7.1f} MB/s", flush=True)
        if bps > best_bps:
            best, best_bps = prefix, bps
    if best_bps <= 0 and not quiet:
        print("  都测不到速度，直接用官方")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        f.write(best)
    if not quiet:
        print(f"  → 用 {best or '官方 docker.io'}")
    return best


if __name__ == "__main__":
    prefix = pick(force="--force" in sys.argv, quiet="--quiet" in sys.argv)
    print(prefix if "--quiet" in sys.argv else f"DOCKER_MIRROR={prefix or '(官方)'}")
