#!/bin/bash
#
# Safe removal of 1,389 duplicate package.json files following PRIME DIRECTIVE
# Zero tolerance for duplicates - removes all non-canonical package.json files
#

set -euo pipefail

DRY_RUN=${1:-false}

# PRIME DIRECTIVE: Define canonical package.json files that MUST be preserved
CANONICAL_FILES=(
    "/c/Users/david/.claude/package.json"
    "/c/Users/david/.claude/hooks/package.json"
    "/c/Users/david/.claude/windows/package.json"
)

log_status() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message"
}

log_status "=== DUPLICATE PACKAGE.JSON CLEANUP SCRIPT ===" "SUCCESS"
log_status "Following PRIME DIRECTIVE: Zero tolerance for duplicates" "SUCCESS"

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "--dry-run" ]]; then
    log_status "DRY RUN MODE: No files will be modified" "WARN"
    DRY_RUN=true
else
    DRY_RUN=false
fi

# Step 1: Verify canonical files exist
log_status "Verifying canonical package.json files exist..." "INFO"
for file in "${CANONICAL_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        log_status "✓ Found canonical file: $file" "SUCCESS"
    else
        log_status "✗ Missing canonical file: $file" "ERROR"
        exit 1
    fi
done

# Step 2: Calculate sizes before cleanup
log_status "=== BEFORE CLEANUP ===" "INFO"
if [[ -d "/c/Users/david/.claude/compiled-servers" ]]; then
    size_gb=$(du -sh "/c/Users/david/.claude/compiled-servers" 2>/dev/null | cut -f1 || echo "Unknown")
    log_status "compiled-servers size: $size_gb" "INFO"
fi
if [[ -d "/c/Users/david/.claude/node_modules" ]]; then
    size_kb=$(du -sh "/c/Users/david/.claude/node_modules" 2>/dev/null | cut -f1 || echo "Unknown")
    log_status "node_modules size: $size_kb" "INFO"
fi

# Step 3: Find all package.json files
log_status "Finding all package.json files..." "INFO"
all_files=$(find "/c/Users/david/.claude" -name "package.json" -type f 2>/dev/null || true)
total_count=$(echo "$all_files" | wc -l)
log_status "Found $total_count total package.json files" "INFO"

# Step 4: Identify duplicates (exclude canonical files)
log_status "Identifying duplicate files..." "INFO"
duplicates=()
canonical_found=()

while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    is_canonical=false
    for canonical in "${CANONICAL_FILES[@]}"; do
        if [[ "$file" == "$canonical" ]]; then
            canonical_found+=("$file")
            is_canonical=true
            break
        fi
    done

    if [[ "$is_canonical" == "false" ]]; then
        # Check if file is in build/cache directories (duplicates)
        if [[ "$file" =~ (node_modules|compiled-servers|dist|build|\.cache|temp|tmp) ]]; then
            duplicates+=("$file")
        else
            log_status "Non-canonical, non-duplicate file found: $file" "WARN"
        fi
    fi
done <<< "$all_files"

log_status "Canonical files found: ${#canonical_found[@]}" "SUCCESS"
log_status "Duplicate files to remove: ${#duplicates[@]}" "INFO"

# Step 5: Create backup list
timestamp=$(date '+%Y%m%d-%H%M%S')
backup_file="/c/Users/david/.claude/duplicate-cleanup-backup-$timestamp.log"

if [[ "$DRY_RUN" == "false" ]]; then
    cat > "$backup_file" << BACKUP_EOF
# Duplicate package.json Cleanup Backup Log
# Generated: $(date '+%Y-%m-%d %H:%M:%S')
# Total files to remove: ${#duplicates[@]}
# Following PRIME DIRECTIVE: Zero tolerance for duplicates

# CANONICAL FILES PRESERVED:
$(printf '# PRESERVED: %s\n' "${CANONICAL_FILES[@]}")

# FILES TO BE REMOVED:
$(printf 'REMOVE: %s\n' "${duplicates[@]}")
BACKUP_EOF
    log_status "Backup list created: $backup_file" "SUCCESS"
else
    log_status "DRY RUN: Would create backup list with ${#duplicates[@]} entries" "WARN"
fi

# Step 6: Remove duplicate files
if [[ ${#duplicates[@]} -eq 0 ]]; then
    log_status "No duplicate files found to remove" "SUCCESS"
    exit 0
fi

if [[ "$DRY_RUN" == "false" ]]; then
    read -p "About to remove ${#duplicates[@]} duplicate files. Continue? (yes/no): " confirmation
    if [[ "$confirmation" != "yes" ]]; then
        log_status "Operation cancelled by user" "WARN"
        exit 0
    fi
fi

log_status "Starting removal of ${#duplicates[@]} duplicate files..." "INFO"

success_count=0
error_count=0
skipped_count=0

for file in "${duplicates[@]}"; do
    # Double-check this is not a canonical file
    is_canonical=false
    for canonical in "${CANONICAL_FILES[@]}"; do
        if [[ "$file" == "$canonical" ]]; then
            log_status "SAFETY CHECK FAILED: Attempted to remove canonical file: $file" "ERROR"
            exit 1
        fi
    done

    if [[ -f "$file" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_status "DRY RUN: Would remove: $file" "WARN"
            ((success_count++))
        else
            if rm -f "$file" 2>/dev/null; then
                ((success_count++))

                # Show progress every 100 files
                if (( success_count % 100 == 0 )); then
                    log_status "Removed $success_count files..." "INFO"
                fi
            else
                log_status "Failed to remove: $file" "ERROR"
                ((error_count++))
                echo "ERROR: Failed to remove $file" >> "$backup_file"
            fi
        fi
    else
        log_status "File not found (already removed?): $file" "WARN"
        ((skipped_count++))
    fi
done

# Step 7: Report results
log_status "=== CLEANUP RESULTS ===" "SUCCESS"
log_status "Files successfully removed: $success_count" "SUCCESS"
log_status "Files with errors: $error_count" $(if [[ $error_count -gt 0 ]]; then echo "ERROR"; else echo "SUCCESS"; fi)
log_status "Files skipped (not found): $skipped_count" "INFO"

if [[ "$DRY_RUN" == "false" ]]; then
    # Step 8: Get directory sizes after cleanup
    log_status "=== AFTER CLEANUP ===" "INFO"
    if [[ -d "/c/Users/david/.claude/compiled-servers" ]]; then
        size_after=$(du -sh "/c/Users/david/.claude/compiled-servers" 2>/dev/null | cut -f1 || echo "Unknown")
        log_status "compiled-servers size after cleanup: $size_after" "INFO"
    fi
    if [[ -d "/c/Users/david/.claude/node_modules" ]]; then
        size_after=$(du -sh "/c/Users/david/.claude/node_modules" 2>/dev/null | cut -f1 || echo "Unknown")
        log_status "node_modules size after cleanup: $size_after" "INFO"
    fi

    # Step 9: Verify canonical files still exist
    log_status "=== VERIFICATION ===" "INFO"
    all_canonical_exist=true
    for file in "${CANONICAL_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            log_status "✓ Canonical file preserved: $file" "SUCCESS"
        else
            log_status "✗ ERROR: Canonical file missing: $file" "ERROR"
            all_canonical_exist=false
        fi
    done

    if [[ "$all_canonical_exist" == "true" ]]; then
        log_status "All canonical files preserved ✓" "SUCCESS"
    else
        log_status "ERROR: Some canonical files were removed!" "ERROR"
        exit 1
    fi

    # Step 10: Final count
    remaining_count=$(find "/c/Users/david/.claude" -name "package.json" -type f 2>/dev/null | wc -l)
    log_status "Remaining package.json files: $remaining_count (should be 3)" "INFO"

    if [[ $remaining_count -eq 3 ]]; then
        log_status "SUCCESS: Cleanup completed - only canonical files remain" "SUCCESS"
    elif [[ $remaining_count -lt 3 ]]; then
        log_status "ERROR: Too few files remaining - canonical files may have been removed" "ERROR"
    else
        log_status "WARNING: More than 3 files remaining - some duplicates may still exist" "WARN"
        log_status "Remaining files:" "INFO"
        find "/c/Users/david/.claude" -name "package.json" -type f 2>/dev/null | while read -r remaining_file; do
            log_status "  - $remaining_file" "INFO"
        done
    fi
fi

log_status "Cleanup script completed successfully" "SUCCESS"
