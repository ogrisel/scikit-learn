# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Run a command under a memory and wall-clock watchdog.

The whole process tree is SIGKILLed when any of the following happens:

- the wall-clock time exceeds --timeout seconds,
- the summed physical footprint of the process tree exceeds --max-footprint-gb
  (on Apple silicon, the footprint includes Metal / GPU allocations made in
  unified memory, unlike RSS),
- the system available memory drops below --min-available-gb.

Peak values are reported on stderr and optionally written as JSON.
"""

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

import psutil

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
RUSAGE_INFO_V2 = 2


class RUsageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


def phys_footprint(pid):
    info = RUsageInfoV2()
    if _libproc.proc_pid_rusage(pid, RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        return 0
    return info.ri_phys_footprint


def tree_pids(proc):
    try:
        return [proc.pid] + [c.pid for c in proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def kill_tree(proc):
    for pid in reversed(tree_pids(proc)):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-footprint-gb", type=float, default=8)
    parser.add_argument("--min-available-gb", type=float, default=6)
    parser.add_argument("--poll", type=float, default=0.1)
    parser.add_argument("--report", default=None)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd

    gb = 1024**3
    start = time.perf_counter()
    popen = subprocess.Popen(cmd)
    proc = psutil.Process(popen.pid)
    peak_fp = peak_rss = 0
    min_avail = psutil.virtual_memory().available
    reason = None
    while popen.poll() is None:
        pids = tree_pids(proc)
        fp = sum(phys_footprint(pid) for pid in pids)
        rss = 0
        for pid in pids:
            try:
                rss += psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:
                pass
        avail = psutil.virtual_memory().available
        peak_fp, peak_rss = max(peak_fp, fp), max(peak_rss, rss)
        min_avail = min(min_avail, avail)
        elapsed = time.perf_counter() - start
        if elapsed > args.timeout:
            reason = f"timeout ({elapsed:.1f}s > {args.timeout}s)"
        elif fp > args.max_footprint_gb * gb:
            reason = f"footprint {fp / gb:.2f} GB > {args.max_footprint_gb} GB"
        elif avail < args.min_available_gb * gb:
            reason = f"system available memory {avail / gb:.2f} GB"
        if reason:
            kill_tree(proc)
            popen.wait()
            break
        time.sleep(args.poll)

    summary = {
        "returncode": popen.returncode,
        "killed": reason,
        "elapsed_s": time.perf_counter() - start,
        "peak_footprint_gb": peak_fp / gb,
        "peak_rss_gb": peak_rss / gb,
        "min_available_gb": min_avail / gb,
    }
    print(f"[watchdog] {json.dumps(summary)}", file=sys.stderr)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(summary, f)
    sys.exit(popen.returncode if reason is None else 137)


if __name__ == "__main__":
    main()
