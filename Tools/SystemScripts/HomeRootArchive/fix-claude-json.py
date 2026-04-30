#!/usr/bin/env python3
"""
Fix Unicode encoding issues in .claude.json file
"""
import json
import shutil
from pathlib import Path

def fix_claude_json():
    claude_json_path = Path(r'C:\Users\david\.claude.json')
    backup_path = Path(r'C:\Users\david\.claude.json.backup')

    print(f"Fixing Unicode encoding in: {claude_json_path}")

    # Create backup
    shutil.copy2(claude_json_path, backup_path)
    print(f"✓ Backup created: {backup_path}")

    # Try different encodings to read the file
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'ascii']

    data = None
    used_encoding = None

    for encoding in encodings:
        try:
            with open(claude_json_path, 'r', encoding=encoding) as f:
                content = f.read()
            # Test if it can be parsed as JSON
            data = json.loads(content)
            used_encoding = encoding
            print(f"✓ Successfully read with encoding: {encoding}")
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"✗ Failed with {encoding}: {type(e).__name__}")
            continue

    if data is None:
        print("❌ Could not read file with any encoding")
        return False

    # Write back with proper UTF-8 encoding
    try:
        with open(claude_json_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✓ File rewritten with UTF-8 encoding")

        # Validate the result
        with open(claude_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print("✓ JSON validation successful")

        # Print some basic info
        if 'projects' in test_data and r'C:\Users\david' in test_data['projects']:
            project = test_data['projects'][r'C:\Users\david']
            if 'mcpServers' in project:
                servers = list(project['mcpServers'].keys())
                print(f"✓ MCP servers found: {servers}")
            else:
                print("ℹ️ No MCP servers found")

        return True

    except Exception as e:
        print(f"❌ Error writing file: {e}")
        # Restore backup
        shutil.copy2(backup_path, claude_json_path)
        print("✓ Backup restored")
        return False

if __name__ == "__main__":
    success = fix_claude_json()
    exit(0 if success else 1)