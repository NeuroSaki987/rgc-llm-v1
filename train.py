import os
import re
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml


# ============================================================
# 路径与默认值
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_META_DIR = TEMP_DIR / "meta"
TEMP_TMP_DIR = TEMP_DIR / "tmp"

DEFAULT_LOCAL_DATASET = PROJECT_ROOT / "data" / "train001.json"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "data" / "train_belle_3_5m.jsonl"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_GENERATED_CONFIG = PROJECT_ROOT / "configs" / "belle_train.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "belle_3_5m"

TRAIN_CMD = ["rgc-train", "fit", "--config"]

RESUME_KEY_CANDIDATES = [
    "resume_from",
    "resume_from_checkpoint",
    "checkpoint",
]



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_atomic(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp_path = TEMP_TMP_DIR / f"{path.name}.tmp"
    ensure_dir(TEMP_TMP_DIR)
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    write_text_atomic(path, text)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"基础配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是 dict: {path}")
    return data


def save_yaml(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = TEMP_TMP_DIR / f"{path.name}.tmp"
    ensure_dir(TEMP_TMP_DIR)
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp_path, path)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.replace("\u3000", " ").replace("\xa0", " ")
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def combine_prompt(instruction: str, inp: str = "") -> str:
    instruction = normalize_text(instruction)
    inp = normalize_text(inp)
    if instruction and inp:
        return f"{instruction}\n\n{inp}"
    return instruction or inp


def is_probably_jsonl(file_path: Path) -> bool:
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return True
    if suffix != ".json":
        return False
    with open(file_path, "r", encoding="utf-8") as f:
        head = f.read(4096).lstrip()
    return not head.startswith("[")


def file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }



# 本地数据读取
def iter_json_array_records(file_path: Path) -> Iterator[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(obj, dict):
        for key in ("data", "records", "items", "train"):
            value = obj.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return

    raise ValueError(f"无法识别的 JSON 顶层结构: {file_path}")


def iter_jsonl_records(file_path: Path) -> Iterator[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{file_path} 第 {line_no} 行不是合法 JSON: {e}") from e
            if isinstance(obj, dict):
                yield obj


def iter_local_records(file_path: Path) -> Iterator[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"数据路径不是文件: {file_path}")

    if is_probably_jsonl(file_path):
        yield from iter_jsonl_records(file_path)
    else:
        yield from iter_json_array_records(file_path)



# 样本转换
# 输出统一为 {"input": "...", "target": "..."}
def convert_record(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    # 1) instruction / input / output
    instruction = normalize_text(row.get("instruction", ""))
    input_text = normalize_text(row.get("input", ""))
    output_text = normalize_text(row.get("output", ""))

    if output_text:
        prompt = combine_prompt(instruction, input_text)
        if prompt:
            return {"input": prompt, "target": output_text}

    # 2) conversations
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        user_msg = ""
        assistant_msg = ""

        for item in conversations:
            if not isinstance(item, dict):
                continue
            speaker = normalize_text(item.get("from", "") or item.get("role", ""))
            value = normalize_text(item.get("value", "") or item.get("content", ""))

            sp = speaker.lower()
            if not user_msg and sp in {"human", "user"} and value:
                user_msg = value
                continue
            if user_msg and not assistant_msg and sp in {"assistant", "gpt", "bot"} and value:
                assistant_msg = value
                break

        if user_msg and assistant_msg:
            return {"input": user_msg, "target": assistant_msg}

    # 3) messages
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        user_msg = ""
        assistant_msg = ""

        for item in messages:
            if not isinstance(item, dict):
                continue
            role = normalize_text(item.get("role", ""))
            content = normalize_text(item.get("content", ""))
            rp = role.lower()

            if not user_msg and rp == "user" and content:
                user_msg = content
                continue
            if user_msg and not assistant_msg and rp == "assistant" and content:
                assistant_msg = content
                break

        if user_msg and assistant_msg:
            return {"input": user_msg, "target": assistant_msg}

    return None


# 转换缓存
def conversion_meta_path(output_jsonl: Path) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(output_jsonl.resolve()))
    return TEMP_META_DIR / f"{safe_name}.meta.json"


def load_conversion_meta(output_jsonl: Path) -> Optional[Dict[str, Any]]:
    meta_path = conversion_meta_path(output_jsonl)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def save_conversion_meta(
    output_jsonl: Path,
    source_path: Path,
    max_samples: int,
    read_count: int,
    written_count: int,
) -> None:
    ensure_dir(TEMP_META_DIR)
    meta = {
        "source": file_signature(source_path),
        "max_samples": max_samples,
        "read_count": read_count,
        "written_count": written_count,
        "output_jsonl": str(output_jsonl.resolve()),
        "saved_at": int(time.time()),
    }
    write_json_atomic(conversion_meta_path(output_jsonl), meta)


def should_skip_conversion(
    source_path: Path,
    output_jsonl: Path,
    max_samples: int,
    force_reconvert: bool,
) -> bool:
    if force_reconvert:
        return False
    if not output_jsonl.exists() or output_jsonl.stat().st_size == 0:
        return False

    meta = load_conversion_meta(output_jsonl)
    if not meta:
        return False

    current_sig = file_signature(source_path)
    return (
        meta.get("source") == current_sig and
        meta.get("max_samples") == max_samples
    )


# 数据转换
def convert_local_dataset_to_jsonl(
    source_path: Path,
    output_jsonl: Path,
    max_samples: int = 0,
    force_reconvert: bool = False,
) -> Tuple[int, int]:
    ensure_dir(output_jsonl.parent)
    ensure_dir(TEMP_TMP_DIR)

    if should_skip_conversion(source_path, output_jsonl, max_samples, force_reconvert):
        print(f"[1/4] 检测到转换结果未过期，跳过重新转换: {output_jsonl}")
        meta = load_conversion_meta(output_jsonl) or {}
        return int(meta.get("read_count", 0)), int(meta.get("written_count", 0))

    if force_reconvert and output_jsonl.exists():
        print(f"[info] 强制重新转换，删除旧文件: {output_jsonl}")
        output_jsonl.unlink()

    print(f"[1/4] 使用本地数据文件: {source_path}")
    print(f"[2/4] 正在转换并写入: {output_jsonl}")

    tmp_output = TEMP_TMP_DIR / f"{output_jsonl.name}.tmp"

    read_count = 0
    written_count = 0
    skipped_count = 0

    with open(tmp_output, "w", encoding="utf-8", newline="\n") as fout:
        for row in iter_local_records(source_path):
            read_count += 1
            converted = convert_record(row)

            if converted is None:
                skipped_count += 1
            else:
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written_count += 1

            if max_samples > 0 and read_count >= max_samples:
                break

            if read_count % 100000 == 0:
                print(
                    f"  progress: read={read_count}, written={written_count}, skipped={skipped_count}"
                )

    if written_count == 0:
        try:
            tmp_output.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(
            "没有成功转换出任何样本。请检查 train001.json 的字段结构是否为 "
            "instruction/input/output 或 conversations 或 messages。"
        )

    os.replace(tmp_output, output_jsonl)
    save_conversion_meta(output_jsonl, source_path, max_samples, read_count, written_count)

    print(
        f"[done] read={read_count}, written={written_count}, skipped={skipped_count}, file={output_jsonl}"
    )
    return read_count, written_count


# checkpoint 续训
def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None

    final_ckpt = output_dir / "final.pt"
    if final_ckpt.exists() and final_ckpt.is_file():
        return final_ckpt

    epoch_items: List[Tuple[int, Path]] = []
    for fp in output_dir.glob("epoch_*.pt"):
        m = re.fullmatch(r"epoch_(\d+)\.pt", fp.name)
        if m:
            epoch_items.append((int(m.group(1)), fp))

    if not epoch_items:
        return None

    epoch_items.sort(key=lambda x: x[0], reverse=True)
    return epoch_items[0][1]


def detect_resume_key(cfg: Dict[str, Any]) -> str:
    training = cfg.get("training")
    if isinstance(training, dict):
        for key in RESUME_KEY_CANDIDATES:
            if key in training:
                return key
    return "resume_from"


# ============================================================
# 训练配置生成
# data.train_file / file_format / text_field / target_field
# training.output_dir
# ============================================================
def build_training_config(
    base_config_path: Path,
    generated_config_path: Path,
    train_jsonl_path: Path,
    output_dir: Path,
    resume_ckpt: Optional[Path],
) -> Path:
    print(f"[3/4] 正在生成训练配置: {generated_config_path}")

    cfg = load_yaml(base_config_path)

    data_cfg = cfg.get("data")
    if data_cfg is None:
        cfg["data"] = {}
    elif not isinstance(data_cfg, dict):
        raise ValueError("基础配置中的 data 必须是 dict")

    training_cfg = cfg.get("training")
    if training_cfg is None:
        cfg["training"] = {}
    elif not isinstance(training_cfg, dict):
        raise ValueError("基础配置中的 training 必须是 dict")

    cfg["data"]["train_file"] = str(train_jsonl_path)
    cfg["data"]["file_format"] = "jsonl"
    cfg["data"]["text_field"] = "input"
    cfg["data"]["target_field"] = "target"

    cfg["training"]["output_dir"] = str(output_dir)

    resume_key = detect_resume_key(cfg)
    if resume_ckpt is not None:
        cfg["training"][resume_key] = str(resume_ckpt)
        print(f"  检测到已有 checkpoint: {resume_ckpt}")
        print(f"  已写入断点续训字段: training.{resume_key}")
    else:
        # 避免旧配置残留错误 resume 路径
        for key in RESUME_KEY_CANDIDATES:
            cfg["training"].pop(key, None)
        print("  未检测到旧 checkpoint，将从头开始训练")

    save_yaml(generated_config_path, cfg)
    return generated_config_path


# 训练启动
def run_training(config_path: Path) -> None:
    cmd = TRAIN_CMD + [str(config_path)]
    print(f"[4/4] 启动训练: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


# 启动前检check
def validate_base_files(dataset_path: Path, base_config_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"本地数据文件不存在: {dataset_path}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"基础配置文件不存在: {base_config_path}")


def cleanup_stale_tmp_files() -> None:
    ensure_dir(TEMP_DIR)
    ensure_dir(TEMP_META_DIR)
    ensure_dir(TEMP_TMP_DIR)

    for tmp_file in TEMP_TMP_DIR.glob("*.tmp"):
        try:
            tmp_file.unlink()
        except Exception:
            pass


# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="将本地 Belle 风格数据转成训练 JSONL，并按照项目文档生成配置后启动训练。"
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_LOCAL_DATASET),
        help="本地数据文件路径，支持 JSON 数组或 JSONL。默认: ./data/train001.json",
    )
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_BASE_CONFIG),
        help="基础训练配置 YAML，默认: ./configs/default.yaml",
    )
    parser.add_argument(
        "--generated-config",
        default=str(DEFAULT_GENERATED_CONFIG),
        help="自动生成的训练配置 YAML，默认: ./configs/belle_train.yaml",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="转换后的训练 JSONL 输出路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="训练输出目录，用于保存 checkpoint",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 表示全量；>0 表示只转换前 N 条样本",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="只做转换与配置生成，不启动训练",
    )
    parser.add_argument(
        "--force-reconvert",
        action="store_true",
        help="强制重新转换 JSONL，不使用旧转换结果",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    base_config_path = Path(args.base_config).resolve()
    generated_config_path = Path(args.generated_config).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()

    cleanup_stale_tmp_files()
    validate_base_files(dataset_path, base_config_path)

    convert_local_dataset_to_jsonl(
        source_path=dataset_path,
        output_jsonl=output_jsonl,
        max_samples=args.max_samples,
        force_reconvert=args.force_reconvert,
    )

    latest_ckpt = find_latest_checkpoint(output_dir)

    generated_cfg = build_training_config(
        base_config_path=base_config_path,
        generated_config_path=generated_config_path,
        train_jsonl_path=output_jsonl,
        output_dir=output_dir,
        resume_ckpt=latest_ckpt,
    )

    if args.skip_train:
        print("已跳过训练，仅完成数据转换与配置生成")
        return

    run_training(generated_cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] 用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n[error] {type(e).__name__}: {e}")
        sys.exit(1)