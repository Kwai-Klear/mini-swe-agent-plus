#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run mini-SWE-agent on SWE-bench instances in batch mode (multiprocessing version)."""

# 参考文档：https://mini-swe-agent.com/latest/usage/swebench/

import json
import random
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Optional
import os

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

# 新增：多进程相关
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# ✅ 新增：为识别 CalledProcessError、做健壮解码与文件操作
import subprocess
import os

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model

from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}

# ---------------------------
# 新增：错误信息工具函数
# ---------------------------
def _safe_text(x) -> Optional[str]:
    """bytes->str 安全解码，避免 None / 解码异常"""
    if x is None:
        return None
    if isinstance(x, bytes):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return x.decode(enc, errors="replace")
            except Exception:
                continue
        return x.decode(errors="replace")
    return str(x)

def _write_error_artifacts(instance_dir: Path, payload: dict):
    """把详细错误写到 {instance}/error.log，便于排查"""
    try:
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "error.log").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    except Exception:
        pass

# 主进程里统一写 preds.json（跨进程无需锁）
def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    output_data = {}
    if output_path.exists():
        output_data = json.loads(output_path.read_text())
    output_data[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }
    output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    if not output_path.exists():
        return
    output_data = json.loads(output_path.read_text())
    if instance_id in output_data:
        del output_data[instance_id]
        output_path.write_text(json.dumps(output_data, indent=2))


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker 不允许双下划线，用占位替代
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


# ---------------------------
# 进程内：轻量进度汇报的 Agent 包装
# ---------------------------
class MPProgressAgent(DefaultAgent):
    """在 step() 时往进程间队列发进度事件，避免跨进程直接持有 UI 对象。"""

    def __init__(self, *args, progress_queue, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_queue = progress_queue
        self.instance_id = instance_id

    def step(self) -> dict:
        try:
            self.progress_queue.put({
                "t": "status",
                "id": self.instance_id,
                "msg": f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
            })
        except Exception:
            # 进度上报失败不应影响任务本身
            pass
        return super().step()


# ---------------------------
# 进程入口函数（必须是顶层可 picklable）
# ---------------------------
def process_instance_proc(
    instance: dict,
    output_dir: str,
    model_name: Optional[str],
    config_path: str | Path,
    progress_queue,
    docker_start_sem=None,  # 可选：控制“启动 Docker”的并发
):
    """单个实例的子进程执行体。返回给主进程用于统一写 preds.json 与收尾 UI。"""
    instance_id = instance["instance_id"]
    instance_dir = Path(output_dir) / instance_id

    # 清理旧轨迹文件，避免半成品
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    # 通知开始
    try:
        progress_queue.put({"t": "start", "id": instance_id})
    except Exception:
        pass

    image_name = get_swebench_docker_image_name(instance)
    config = yaml.safe_load(get_config_path(config_path).read_text())
    model = get_model(model_name, config=config.get("model", {}))
    task = instance["problem_statement"]

    agent = None
    extra_info = None

    # （强烈推荐）限制“拉起/连接 Docker 环境”的并发数
    try:
        progress_queue.put({"t": "status", "id": instance_id, "msg": "Pulling/starting docker (queued)"})
    except Exception:
        pass

    if docker_start_sem is not None:
        docker_start_sem.acquire()

    try:
        try:
            try:
                progress_queue.put({"t": "status", "id": instance_id, "msg": "Pulling/starting docker"})
            except Exception:
                pass
            env = DockerEnvironment(**(config.get("environment", {}) | {"image": image_name}))
        finally:
            if docker_start_sem is not None:
                docker_start_sem.release()
                
        # 真正跑 agent
        agent = MPProgressAgent(
            model,
            env,
            progress_queue=progress_queue,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task)

    except Exception as e:
        # 捕获所有异常，保证主进程能收到结果并收尾
        exit_status = type(e).__name__

        if isinstance(e, subprocess.CalledProcessError):
            # 最大化还原子进程报错信息
            err_payload = {
                "type": "CalledProcessError",
                "returncode": e.returncode,
                "cmd": e.cmd,
                "stdout": _safe_text(getattr(e, "stdout", None)),
                "stderr": _safe_text(getattr(e, "stderr", None)),
                "traceback": traceback.format_exc(),
            }
            # 方便在 preds.json 里直观看到关键信息
            result = (
                f"[CalledProcessError rc={e.returncode}] cmd={e.cmd}\n"
                f"--- STDOUT ---\n{err_payload['stdout'] or ''}\n"
                f"--- STDERR ---\n{err_payload['stderr'] or ''}"
            )
            extra_info = err_payload
            # 同时落盘一个 error.log
            _write_error_artifacts(instance_dir, err_payload)

            # 顺带发一条短状态到 UI
            try:
                short = (err_payload["stderr"] or err_payload["stdout"] or "").strip().splitlines()
                head = short[0][:180] if short else ""
                progress_queue.put({"t": "status", "id": instance_id,
                                   "msg": f"CalledProcessError rc={e.returncode} · {head}"})
            except Exception:
                pass
        else:
            # 非 CalledProcessError 保持原逻辑，但加更可读的回溯
            tb_str = "".join(traceback.TracebackException.from_exception(e).format())
            result = f"{type(e).__name__}: {e}\n{tb_str}"
            extra_info = {"traceback": tb_str}

    finally:
        # 各自进程只写自己的 traj 文件（互不干扰）
        try:
            save_traj(
                agent,
                instance_dir / f"{instance_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                instance_id=instance_id,
            )
        except Exception:
            # 轨迹保存失败不阻断总体结果返回
            pass

    # 返回给主进程做统一写 preds.json 与 UI 收尾
    model_name_resolved = getattr(getattr(model, "config", None), "model_name", str(model))
    return {
        "instance_id": instance_id,
        "exit_status": exit_status,
        "result": result,
        "model_name": model_name_resolved,
    }


# ---------------------------
# 过滤与切片（原样保留）
# ---------------------------
def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    if filter_spec:
        instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        print(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            print(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# ---------------------------
# 主进程：消费进度事件（Queue -> progress_manager）
# ---------------------------
def pump_progress_events(q, progress_manager: RunBatchProgressManager, stop_token: str):
    """在单独线程中消费子进程上报的事件，并更新 UI。"""
    while True:
        try:
            ev = q.get()
        except (EOFError, OSError):
            break
        if not ev:
            continue
        if isinstance(ev, dict) and ev.get("t") == stop_token:
            break
        t = ev.get("t")
        iid = ev.get("id")
        if t == "start":
            progress_manager.on_instance_start(iid)
        elif t == "status":
            progress_manager.update_instance_status(iid, ev.get("msg", ""))
        # 结束事件由主进程在拿到结果后调用，不在这里处理


# ---------------------------
# CLI 入口
# ---------------------------
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset"),
    split: str = typer.Option("dev", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker processes"),
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Model to use"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances"),
    config: Path = typer.Option(
        builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file"
    ),
    docker_start_concurrency: int = typer.Option(
        8, "--docker-start-concurrency",
        help="Max concurrent 'start docker environment' ops to avoid daemon/disk thrash",
    ),
) -> None:
    # 加载数据集
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset {dataset_path}, split {split}...")
    if subset in ('verified100', 'verified236', 'mysmith'):
        instances = list(load_dataset("json", data_files=dataset_path, split="train"))
    else:
        instances = list(load_dataset(dataset_path, split=split))

    
    if 'checkpoint' in model or 'hf_bin' in model:
        # 本地访问时清空代理环境变量
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("https_proxy", None)

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 跳过已完成
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        print(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    else:
        # redo 的场景中，预先清掉将要跑的实例的历史记录，避免不一致
        for inst in instances:
            remove_from_preds_file(output_path / "preds.json", inst["instance_id"])

    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    # 进程间通信：Manager.Queue 用于进度事件；BoundedSemaphore 用于限制 Docker 启动并发
    ctx = mp.get_context("spawn")  # 用 spawn 更干净，避免 fork 遗留状态
    mgr = ctx.Manager()
    progress_queue = mgr.Queue()
    stop_token = "__STOP__"
    docker_sem = mgr.BoundedSemaphore(docker_start_concurrency if docker_start_concurrency > 0 else 1)

    # 独立线程消费进度事件（避免阻塞主线程）
    pump_thread = threading.Thread(
        target=pump_progress_events, args=(progress_queue, progress_manager, stop_token), daemon=True
    )
    pump_thread.start()

    # Live 渲染 + 进程池执行
    with Live(progress_manager.render_group, refresh_per_second=4):
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            future_to_id = {
                executor.submit(
                    process_instance_proc,
                    instance,
                    str(output_path),
                    model,
                    config,
                    progress_queue,
                    docker_sem
                ): instance["instance_id"]
                for instance in instances
            }

            try:
                for fut in as_completed(future_to_id):
                    iid = future_to_id[fut]
                    try:
                        payload = fut.result()
                        # 统一写 preds.json（主进程单写者，避免跨进程文件锁）
                        update_preds_file(
                            output_path / "preds.json",
                            payload["instance_id"],
                            payload["model_name"],
                            payload["result"],
                        )
                        # UI 收尾
                        progress_manager.on_instance_end(iid, payload["exit_status"])

                        # ✅ 可见性提示：遇到 CalledProcessError 时，引导查看 error.log
                        if payload.get("exit_status") == "CalledProcessError":
                            print(f"[{iid}] CalledProcessError · 详情见 {(output_path / iid / 'error.log')}")
                    except Exception as e:
                        print(f"Error in future for instance {iid}: {e}")
                        traceback.print_exc()
                        progress_manager.on_uncaught_exception(iid, e)
            except KeyboardInterrupt:
                print("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for fut in future_to_id:
                    if not fut.running() and not fut.done():
                        fut.cancel()
                # 尽量把已在跑的收尾
                try:
                    for fut in as_completed(future_to_id):
                        pass
                except Exception:
                    pass

    # 停止进度泵
    try:
        progress_queue.put({"t": stop_token})
    except Exception:
        pass
    pump_thread.join(timeout=5.0)


if __name__ == "__main__":
    app()
