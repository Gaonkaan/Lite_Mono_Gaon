import os, shutil
from pathlib import Path

SRC = Path(r"E:\kitti_data")  # 원본 KITTI
DST = Path(r"E:\kitti_mini")  # 만들 mini KITTI
SPLIT = Path(r"C:\Users\권가온\Lite-Mono\splits\eigen_zhou\train_files.txt")

# 용량/시간 줄이려면 N을 줄이면 됨 (예: 5000, 10000, 20000)
N = 4000

# jpg/png 자동 감지: 둘 중 존재하는 확장자를 찾아서 복사
EXT_CANDIDATES = [".jpg", ".png"]

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def copy_if_exists(rel_path_no_ext: str):
    for ext in EXT_CANDIDATES:
        src = SRC / (rel_path_no_ext + ext)
        if src.exists():
            dst = DST / (rel_path_no_ext + ext)
            ensure_dir(dst)
            if not dst.exists():
                shutil.copy2(src, dst)
            return True
    return False

def main():
    if not SPLIT.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT}")

    lines = SPLIT.read_text(encoding="utf-8").strip().splitlines()
    lines = lines[:N]

    miss = 0
    copied = 0

    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue

        folder = parts[0]           # 예: 2011_09_26/2011_09_26_drive_0001_sync
        frame = int(parts[1])       # 예: 123
        side = parts[2]             # l or r

        cam = "image_02" if side == "l" else "image_03"

        # monodepth2/lite-mono 경로 관례: {folder}/{cam}/data/{frame:010d}.(jpg/png)
        def rel_no_ext(fidx: int):
            return f"{folder}/{cam}/data/{fidx:010d}"

        # 중심 프레임 + (t-1, t+1)도 같이 복사 (self-supervised에서 쓰는 경우가 많음)
        for d in (0, -1, +1):
            ok = copy_if_exists(rel_no_ext(frame + d))
            if ok:
                copied += 1
            else:
                miss += 1

    # splits 복사 (colab에서 같이 쓰기 편하게)
    split_dst = DST / "splits" / "eigen_zhou"
    split_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SPLIT, split_dst / "train_files.txt")

    print("Done.")
    print("Mini KITTI:", DST)
    print("Copied files:", copied)
    print("Missing files (not found):", miss)
    print("Tip: if missing is huge, your KITTI folder structure or extension differs.")

if __name__ == "__main__":
    main()
