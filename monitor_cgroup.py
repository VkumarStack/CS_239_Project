#!/usr/bin/env python3
"""Monitor cgroup memory usage in real-time."""

import time
import sys
from pathlib import Path

def read_cgroup_memory() -> dict:
    """Read current cgroup memory statistics."""
    cgroup_path = Path("/sys/fs/cgroup/chromabench")
    
    stats = {}
    
    # Read memory.max
    max_file = cgroup_path / "memory.max"
    if max_file.exists():
        value = max_file.read_text().strip()
        stats['max'] = int(value) if value != "max" else None
    
    # Read memory.current
    current_file = cgroup_path / "memory.current"
    if current_file.exists():
        stats['current'] = int(current_file.read_text().strip())
    
    # Read memory.events (OOM kills, etc)
    events_file = cgroup_path / "memory.events"
    if events_file.exists():
        events = {}
        for line in events_file.read_text().strip().split('\n'):
            if line:
                key, val = line.split()
                events[key] = int(val)
        stats['events'] = events
    
    # Read memory.stat  
    stat_file = cgroup_path / "memory.stat"
    if stat_file.exists():
        mem_stats = {}
        for line in stat_file.read_text().strip().split('\n'):
            if line:
                key, val = line.split()
                mem_stats[key] = int(val)
        stats['stats'] = mem_stats
    
    return stats

def format_bytes(b: int | None) -> str:
    """Format bytes in human-readable form."""
    if b is None:
        return "unlimited"
    if b >= (1 << 30):
        return f"{b / (1 << 30):.2f} GiB"
    if b >= (1 << 20):
        return f"{b / (1 << 20):.2f} MiB"
    return f"{b / (1 << 10):.2f} KiB"

def main():
    print("Monitoring cgroup memory usage. Press Ctrl+C to stop.\n")
    print(f"{'Time':<8} {'Current':<12} {'Limit':<12} {'Usage%':<8} {'File':<12} {'Anon':<12} {'OOM Kills':<10}")
    print("-" * 90)
    
    last_oom = 0
    
    try:
        while True:
            stats = read_cgroup_memory()
            
            current = stats.get('current', 0)
            max_mem = stats.get('max')
            
            usage_pct = (current / max_mem * 100) if max_mem else 0
            
            # Get file vs anonymous memory
            mem_stats = stats.get('stats', {})
            file_mem = mem_stats.get('file', 0)
            anon_mem = mem_stats.get('anon', 0)
            
            # Check for OOM kills
            events = stats.get('events', {})
            oom_kill = events.get('oom_kill', 0)
            oom_indicator = f" [+{oom_kill - last_oom}]" if oom_kill > last_oom else ""
            last_oom = oom_kill
            
            print(f"{time.strftime('%H:%M:%S'):<8} "
                  f"{format_bytes(current):<12} "
                  f"{format_bytes(max_mem):<12} "
                  f"{usage_pct:>6.1f}%  "
                  f"{format_bytes(file_mem):<12} "
                  f"{format_bytes(anon_mem):<12} "
                  f"{oom_kill:<10}{oom_indicator}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")

if __name__ == "__main__":
    main()
