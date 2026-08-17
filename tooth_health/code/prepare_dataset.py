"""
全自动处理流程：扫`tooth_health/data/`下所有Label Studio导出的zip
（"YOLO with Images"格式，文件名类似`project-66-at-2026-08-17-...zip`），
自动解压+合并成统一的YOLO训练集。**每次有新project导出的zip就直接扔进
`tooth_health/data/`，重跑这一个脚本就行，不用管有几个zip、不用手动解压、
不用记得上次跑到哪——脚本自己发现所有zip、自己判断哪些是新增/更新过的、
自己重建完整数据集。**

用法（以后每次补充新数据都是这一条命令）：
    python3 tooth_health/code/prepare_dataset.py

处理逻辑：
  1. 扫`--zips_dir`（默认`tooth_health/data/`）下所有*.zip
  2. 每个zip解压到`raw_exports/<zip文件名(不含扩展名)>/`，用zip文件的
     大小+修改时间判断有没有变化，没变化就跳过重新解压（避免每次全量
     重复解压，zip不多的时候其实无所谓，但数据以后可能越攒越多）
  3. 兼容Label Studio zip两种可能的目录结构：images/labels/classes.txt
     直接在zip根目录，或者外面包一层文件夹——自动探测，不用你关心导出
     zip内部具体长什么样
  4. 解压完，把`raw_exports/`下所有子目录一起交给合并逻辑（复用
     merge_labelstudio_yolo_exports.py的核心函数），重建
     `yolo_dataset/`（每次全量重建，不是增量追加——数据集这个体量下
     全量重建更简单可靠，不用担心"部分更新导致train/val分布跟之前
     不一致"这种问题）
"""
import argparse
import zipfile
from pathlib import Path

from merge_labelstudio_yolo_exports import merge_exports

MARKER_NAME = ".extracted_from"


def _zip_fingerprint(zip_path: Path) -> str:
    stat = zip_path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _find_data_root(extract_dir: Path) -> Path:
    """Label Studio的zip有时是images/labels/classes.txt直接在根目录，
    有时外面包一层文件夹——探测哪一层才是真正装着这三样东西的目录。"""
    if (extract_dir / "classes.txt").exists():
        return extract_dir
    candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
    for c in candidates:
        if (c / "classes.txt").exists():
            return c
    raise FileNotFoundError(
        f"{extract_dir} 解压后找不到classes.txt（不管是根目录还是子目录），"
        "确认导出格式选的是'YOLO with Images'")


def extract_zips(zips_dir: Path, raw_exports_dir: Path) -> list:
    raw_exports_dir.mkdir(parents=True, exist_ok=True)
    export_dirs = []
    zip_files = sorted(zips_dir.glob("*.zip"))
    if not zip_files:
        print(f"[警告] {zips_dir} 下没有找到任何zip文件")
        return []

    for zip_path in zip_files:
        name = zip_path.stem
        target_dir = raw_exports_dir / name
        marker = target_dir / MARKER_NAME
        fingerprint = _zip_fingerprint(zip_path)

        if marker.exists() and marker.read_text().strip() == fingerprint:
            print(f"  {zip_path.name}: 未变化，跳过重新解压")
        else:
            if target_dir.exists():
                import shutil
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target_dir)
            marker.write_text(fingerprint)
            print(f"  {zip_path.name}: 已解压到 {target_dir}")

        data_root = _find_data_root(target_dir)
        export_dirs.append(data_root)

    return export_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips_dir", default="tooth_health/data",
                    help="放Label Studio导出zip的目录（默认tooth_health/data，"
                         "以后每次导出新project直接把zip扔这里）")
    ap.add_argument("--raw_exports_dir", default="tooth_health/data/raw_exports")
    ap.add_argument("--out_dir", default="tooth_health/data/yolo_dataset")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    zips_dir = Path(args.zips_dir)
    raw_exports_dir = Path(args.raw_exports_dir)

    print(f"扫描 {zips_dir} 下的zip文件...")
    export_dirs = extract_zips(zips_dir, raw_exports_dir)
    if not export_dirs:
        print("没有可处理的数据，退出")
        return

    print(f"\n共 {len(export_dirs)} 个project导出，开始合并...")
    merge_exports(export_dirs, Path(args.out_dir), args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
