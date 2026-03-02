#!/usr/bin/env python3
"""Check if process is running under cgroup memory limits."""

import os
import sys

def parse_size(value_str: str) -> int | None:
    """Parse memory size from string (handles 'max', bytes, etc)."""
    value_str = value_str.strip()
    if value_str == "max":
        return None
    try:
        return int(value_str)
    except ValueError:
        return None

def format_bytes(b: int | None) -> str:
    """Format bytes in human-readable form."""
    if b is None:
        return "unlimited"
    if b >= (1 << 40):
        return f"{b / (1 << 40):.2f} TiB"
    if b >= (1 << 30):
        return f"{b / (1 << 30):.2f} GiB"
    if b >= (1 << 20):
        return f"{b / (1 << 20):.2f} MiB"
    if b >= (1 << 10):
        return f"{b / (1 << 10):.2f} KiB"
    return f"{b} bytes"

def get_cgroup_version() -> str:
    """Detect cgroup version."""
    if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        return "v2"
    elif os.path.exists("/sys/fs/cgroup/memory"):
        return "v1"
    else:
        return "unknown"

def check_cgroup_v2() -> dict:
    """Check cgroup v2 memory limits."""
    info = {"version": "v2", "in_cgroup": False, "limit": None, "current": None, "path": None}
    
    try:
        # Find which cgroup this process belongs to
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3 and parts[0] == "0":
                    info["path"] = parts[2]
                    break
        
        if not info["path"]:
            return info
        
        # Construct the cgroup path
        cgroup_path = f"/sys/fs/cgroup{info['path']}"
        info["in_cgroup"] = True
        
        # Read memory.max
        try:
            with open(f"{cgroup_path}/memory.max", "r") as f:
                info["limit"] = parse_size(f.read())
        except FileNotFoundError:
            pass
        
        # Read memory.current
        try:
            with open(f"{cgroup_path}/memory.current", "r") as f:
                info["current"] = parse_size(f.read())
        except FileNotFoundError:
            pass
            
    except Exception as e:
        info["error"] = str(e)
    
    return info

def check_cgroup_v1() -> dict:
    """Check cgroup v1 memory limits."""
    info = {"version": "v1", "in_cgroup": False, "limit": None, "current": None, "path": None}
    
    try:
        # Find which memory cgroup this process belongs to
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 3 and "memory" in parts[1]:
                    info["path"] = parts[2]
                    break
        
        if not info["path"]:
            return info
        
        # Construct the cgroup path
        cgroup_path = f"/sys/fs/cgroup/memory{info['path']}"
        info["in_cgroup"] = True
        
        # Read memory.limit_in_bytes
        try:
            with open(f"{cgroup_path}/memory.limit_in_bytes", "r") as f:
                limit = parse_size(f.read())
                # v1 uses a very large value to indicate unlimited
                if limit and limit < (1 << 62):
                    info["limit"] = limit
        except FileNotFoundError:
            pass
        
        # Read memory.usage_in_bytes
        try:
            with open(f"{cgroup_path}/memory.usage_in_bytes", "r") as f:
                info["current"] = parse_size(f.read())
        except FileNotFoundError:
            pass
            
    except Exception as e:
        info["error"] = str(e)
    
    return info

def get_system_memory() -> int:
    """Get total system memory."""
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb * 1024
    return 0

def main():
    print("=== Cgroup Memory Limit Check ===\n")
    
    print(f"Process PID: {os.getpid()}")
    print(f"Cgroup Version: {get_cgroup_version()}\n")
    
    # Check cgroup info based on version
    cgroup_version = get_cgroup_version()
    
    if cgroup_version == "v2":
        info = check_cgroup_v2()
    elif cgroup_version == "v1":
        info = check_cgroup_v1()
    else:
        print("ERROR: Could not detect cgroup version")
        sys.exit(1)
    
    # Display cgroup information
    if "error" in info:
        print(f"Error reading cgroup info: {info['error']}")
        sys.exit(1)
    
    print(f"Cgroup Path: {info['path'] or 'Not found'}")
    print(f"In Memory Cgroup: {info['in_cgroup']}")
    print(f"Memory Limit: {format_bytes(info['limit'])}")
    print(f"Current Usage: {format_bytes(info['current'])}")
    
    # Compare with system memory
    system_mem = get_system_memory()
    print(f"\nSystem Total Memory: {format_bytes(system_mem)}")
    
    # Verdict
    print("\n=== Analysis ===")
    
    if not info['in_cgroup']:
        print("❌ Process is NOT in a memory cgroup")
        print("   The process will see system-wide memory limits only.")
    elif info['limit'] is None:
        print("⚠️  Process is in a cgroup but has NO memory limit")
        print("   The cgroup is not constraining memory usage.")
    elif info['limit'] >= system_mem * 0.95:
        print("⚠️  Cgroup limit is very close to system memory")
        print(f"   Limit: {format_bytes(info['limit'])} vs System: {format_bytes(system_mem)}")
        print("   The cgroup is unlikely to trigger before system-wide pressure.")
    else:
        print("✓ Process IS constrained by cgroup memory limit")
        print(f"  Limit: {format_bytes(info['limit'])} ({info['limit'] / system_mem * 100:.1f}% of system memory)")
        if info['current']:
            print(f"  Current: {format_bytes(info['current'])} ({info['current'] / info['limit'] * 100:.1f}% of limit)")
        print("\n  Your benchmark should use this limit in calculations.")
    
    # Provide recommendation
    if info['limit'] and info['limit'] < system_mem:
        print(f"\n✓ get_mem_total_bytes() should return: {format_bytes(info['limit'])}")
    else:
        print(f"\n✓ get_mem_total_bytes() should return: {format_bytes(system_mem)}")

if __name__ == "__main__":
    main()
