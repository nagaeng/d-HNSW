#!/usr/bin/env python3
"""
Simple validation script to verify the trigger condition changes
Checks that the modified code uses <= instead of < for overflow detection
"""

import os
import re

def check_trigger_conditions():
    """Check that overflow trigger conditions were changed from < to <="""

    # File to check
    cpp_file = "src/dhnsw/DistributedHnsw.cpp"

    if not os.path.exists(cpp_file):
        print(f"ERROR: {cpp_file} not found")
        return False

    with open(cpp_file, 'r') as f:
        content = f.read()

    # Patterns to check - we changed these from < to <=
    patterns_to_check = [
        r'if\(total_new_levels \* sizeof\(int\) <= available_levels_bytes\)',
        r'if\(total_new_offsets \* sizeof\(size_t\) <= available_offsets_bytes\)',
        r'if\(region_len \* sizeof\(faiss::storage_idx_t\) <= available_neighbors_bytes\)',
        r'if \(append_size_bytes <= available_space\)',
        r'if \(total_new_data_bytes <= available_internal_gap_bytes\)'
    ]

    print("Checking trigger condition changes...")
    all_correct = True

    for pattern in patterns_to_check:
        if re.search(pattern, content):
            print(f"✓ Found: {pattern}")
        else:
            print(f"✗ Missing: {pattern}")
            all_correct = False

    # Check that old patterns are NOT present
    old_patterns = [
        r'if\(total_new_levels \* sizeof\(int\) < available_levels_bytes\)',
        r'if\(total_new_offsets \* sizeof\(size_t\) < available_offsets_bytes\)',
        r'if\(region_len \* sizeof\(faiss::storage_idx_t\) < available_neighbors_bytes\)',
        r'if \(append_size_bytes < available_space\)',
        r'if \(total_new_data_bytes < available_internal_gap_bytes\)'
    ]

    print("\nChecking that old conditions were removed...")
    for pattern in old_patterns:
        if re.search(pattern, content):
            print(f"✗ Still found old pattern: {pattern}")
            all_correct = False
        else:
            print(f"✓ Old pattern removed: {pattern}")

    return all_correct

def check_throughput_logger():
    """Check that ThroughputLogger has event recording capabilities"""

    hh_file = "src/dhnsw/reconstruction.hh"
    cpp_file = "src/dhnsw/reconstruction.cpp"

    if not os.path.exists(hh_file) or not os.path.exists(cpp_file):
        print(f"ERROR: reconstruction files not found")
        return False

    with open(hh_file, 'r') as f:
        hh_content = f.read()

    with open(cpp_file, 'r') as f:
        cpp_content = f.read()

    checks = [
        ('record_event method in header', r'void record_event\(const std::string& event\);', hh_content),
        ('event field in ThroughputSample', r'std::string event;', hh_content),
        ('record_event implementation', r'void ThroughputLogger::record_event', cpp_content),
        ('event in CSV output', r'<< sample\.event <<', cpp_content)
    ]

    print("\nChecking ThroughputLogger event recording...")
    all_correct = True

    for name, pattern, content in checks:
        if re.search(pattern, content):
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            all_correct = False

    return all_correct

def check_lsh_cache_fix():
    """Check that LSHCache deserialization was fixed"""

    hh_file = "src/dhnsw/reconstruction.hh"
    cpp_file = "src/dhnsw/reconstruction.cpp"

    if not os.path.exists(hh_file) or not os.path.exists(cpp_file):
        print(f"ERROR: reconstruction files not found")
        return False

    with open(hh_file, 'r') as f:
        hh_content = f.read()

    with open(cpp_file, 'r') as f:
        cpp_content = f.read()

    checks = [
        ('deserialize_into method signature', r'static void deserialize_into.*LSHCache&', hh_content),
        ('reinitialize method', r'void reinitialize\(.*new_dim.*new_num_tables.*new_num_bits', hh_content),
        ('deserialize_into implementation', r'void LSHCache::deserialize_into', cpp_content),
        ('reinitialize implementation', r'void LSHCache::reinitialize', cpp_content),
        ('sync_insert_cache uses deserialize_into', r'LSHCache::deserialize_into\(cache_data, insert_cache_\)', cpp_content)
    ]

    print("\nChecking LSHCache deserialization fix...")
    all_correct = True

    for name, pattern, content in checks:
        if re.search(pattern, content):
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            all_correct = False

    return all_correct

def main():
    print("=" * 60)
    print("VALIDATION: Reconstruction Trigger Modifications")
    print("=" * 60)

    checks = [
        ("Trigger condition changes (< to <=)", check_trigger_conditions),
        ("ThroughputLogger event recording", check_throughput_logger),
        ("LSHCache deserialization fix", check_lsh_cache_fix)
    ]

    all_passed = True

    for name, check_func in checks:
        print(f"\n{'-' * 20} {name} {'-' * 20}")
        if check_func():
            print(f"✓ {name}: PASSED")
        else:
            print(f"✗ {name}: FAILED")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("The reconstruction trigger modifications are correctly implemented.")
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("Please review the implementation.")

    print("=" * 60)

    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
