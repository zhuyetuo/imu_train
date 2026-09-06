"""
日志：同时写终端和文件。文件在 LABEL_LOG_DIR（默认 label_service/logs/）下：
  label_service.log   服务自己的日志：启动参数、每次推理（哪个文件、耗时、各类别几段）、
                      训练任务提交/完成/失败、报错堆栈
  access.log          uvicorn 的 HTTP 访问日志（谁在什么时候调了什么接口、状态码）
按天切、各留 14 天，文件名带日期后缀（label_service.log.2026-09-06）。
推理在子进程里跑，子进程的日志也走同一个文件（多进程追加写同一个文件，
单条记录不会撕裂，够用）。
"""

import logging
import logging.handlers
import os

from label_service import config

_FMT = "%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s"


def _file_handler(name: str) -> logging.Handler:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    h = logging.handlers.TimedRotatingFileHandler(
        os.path.join(config.LOG_DIR, name), when="midnight", backupCount=14, encoding="utf-8",
    )
    h.setFormatter(logging.Formatter(_FMT))
    return h


def setup_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_label_service_configured", False):
        return
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter(_FMT))
    root.addHandler(stream)
    root.addHandler(_file_handler("label_service.log"))

    # uvicorn 的访问日志单独一个文件，不跟业务日志混；它默认自己往终端打，
    # 这里只追加文件 handler
    access = logging.getLogger("uvicorn.access")
    access.addHandler(_file_handler("access.log"))
    root._label_service_configured = True  # type: ignore[attr-defined]


def worker_setup_logging() -> None:
    """推理子进程里调：只写文件，不再往终端重复打（主进程已经打了启动信息）"""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    root.addHandler(_file_handler("label_service.log"))
