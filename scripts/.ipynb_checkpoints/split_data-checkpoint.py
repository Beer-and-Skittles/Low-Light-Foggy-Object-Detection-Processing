# <Project>/scripts/split_data.py
import argparse, random, os
from pathlib import Path
import yaml

from datasets.exdark import list_exdark_pairs
from datasets.rtts import list_rtts_pairs


def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def write_split_list(pairs, out_txt):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for img_path, anno_path in pairs:
            # store absolute paths or relative paths — choose one:
            # I recommend absolute to avoid cwd headaches.
            img = str(img_path.resolve())
            anno = "" if anno_path is None else str(Path(anno_path).resolve())
            f.write(f"{img}\t{anno}\n")


def split_pairs(pairs, train_ratio, seed):
    random.seed(seed)
    pairs = pairs[:]  # copy
    random.shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    return pairs[:n_train], pairs[n_train:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="../configs/default.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)

    proj_root = Path(__file__).resolve().parent.parent  # <Project>/
    data_root = (proj_root / cfg["data_root"]).resolve()

    split_dir = proj_root / cfg["output"]["split_dir"]

    train_ratio = cfg["splits"]["train_ratio"]
    seed = cfg["splits"]["seed"]

    all_train = []
    all_test = []

    # -------- ExDark --------
    if cfg["splits"]["exdark"]["include"]:
        exdark_img_root = data_root / cfg["paths"]["exdark_images"]
        exdark_anno_root = data_root / cfg["paths"]["exdark_annos"]
        exdark_pairs = list_exdark_pairs(exdark_img_root, exdark_anno_root)
        
        ex_train, ex_test = split_pairs(exdark_pairs, train_ratio, seed)
        write_split_list(ex_train, split_dir / "exdark_train.txt")
        write_split_list(ex_test,  split_dir / "exdark_test.txt")

        all_train += ex_train
        all_test  += ex_test

        print(f"[ExDark] total={len(exdark_pairs)} train={len(ex_train)} test={len(ex_test)}")

    # -------- RTTS --------
    if cfg["splits"]["rtts"]["include"]:
        rtts_img_root = data_root / cfg["paths"]["rtts_images"]
        rtts_anno_root = data_root / cfg["paths"]["rtts_annos"]

        rtts_pairs = list_rtts_pairs(rtts_img_root, rtts_anno_root)
        rt_train, rt_test = split_pairs(rtts_pairs, train_ratio, seed)

        write_split_list(rt_train, split_dir / "rtts_train.txt")
        write_split_list(rt_test,  split_dir / "rtts_test.txt")

        all_train += rt_train
        all_test  += rt_test

        print(f"[RTTS] total={len(rtts_pairs)} train={len(rt_train)} test={len(rt_test)}")

    # -------- Combined (optional) --------
    write_split_list(all_train, split_dir / "combined_train.txt")
    write_split_list(all_test,  split_dir / "combined_test.txt")
    print(f"[Combined] train={len(all_train)} test={len(all_test)}")
    print(f"Split lists written to: {split_dir.resolve()}")


if __name__ == "__main__":
    main()
