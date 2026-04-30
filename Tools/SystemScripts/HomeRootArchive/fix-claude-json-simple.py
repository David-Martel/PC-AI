#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

def fix_claude_json():
    claude_json_path = Path(r'C:\Users\david\.claude.json')
    backup_path = Path(r'C:\Users\david\.claude.json.backup')

    print(f"Fixing Unicode encoding in: {claude_json_path}")

    # Create backup
    shutil.copy2(claude_json_path, backup_path)
    print(f"[OK] Backup created: {backup_path}")

    # Try different encodings to read the file
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']

    data = None
    used_encoding = None

    for encoding in encodings:
        try:
            with open(claude_json_path, 'r', encoding=encoding) as f:
                content = f.read()
            # Clean any problematic characters
            content = content.replace('\x8f', '').replace('\x81', '').replace('\x8d', '').replace('\x90', '').replace('\x9d', '')
            # Test if it can be parsed as JSON
            data = json.loads(content)
            used_encoding = encoding
            print(f"[OK] Successfully read with encoding: {encoding}")
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"[FAIL] Failed with {encoding}: {type(e).__name__}")
            continue

    if data is None:
        print("[ERROR] Could not read file with any encoding")
        return False

    # Write back with proper UTF-8 encoding
    try:
        with open(claude_json_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
        print("[OK] File rewritten with UTF-8 encoding")

        # Validate the result
        with open(claude_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print("[OK] JSON validation successful")

        return True

    except Exception as e:
        print(f"[ERROR] Error writing file: {e}")
        # Restore backup
        shutil.copy2(backup_path, claude_json_path)
        print("[OK] Backup restored")
        return False

if __name__ == "__main__":
    success = fix_claude_json()
    exit(0 if success else 1)