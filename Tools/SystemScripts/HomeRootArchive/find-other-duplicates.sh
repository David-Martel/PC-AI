#!/bin/bash
#
# Find other duplicate file patterns in .claude directory
# Following PRIME DIRECTIVE: Zero tolerance for duplicates
#

set -euo pipefail

log_status() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message"
}

log_status "=== FINDING OTHER DUPLICATE PATTERNS ===" "SUCCESS"

# Check for multiple .gitignore files
log_status "Checking for duplicate .gitignore files..." "INFO"
gitignore_count=$(find "/c/Users/david/.claude" -name ".gitignore" -type f 2>/dev/null | wc -l)
if [[ $gitignore_count -gt 1 ]]; then
    log_status "Found $gitignore_count .gitignore files:" "WARN"
    find "/c/Users/david/.claude" -name ".gitignore" -type f 2>/dev/null | while read -r file; do
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        log_status "  - $file ($size bytes)" "INFO"
    done
else
    log_status "✓ Only one .gitignore found" "SUCCESS"
fi

# Check for multiple README files
log_status "Checking for duplicate README files..." "INFO"
readme_files=$(find "/c/Users/david/.claude" -name "README*" -type f 2>/dev/null | grep -v node_modules | grep -v compiled-servers || true)
readme_count=$(echo "$readme_files" | grep -v '^$' | wc -l)
if [[ $readme_count -gt 1 ]]; then
    log_status "Found $readme_count README files:" "WARN"
    echo "$readme_files" | while read -r file; do
        [[ -z "$file" ]] && continue
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        log_status "  - $file ($size bytes)" "INFO"
    done
else
    log_status "✓ Single README file found" "SUCCESS"
fi

# Check for duplicate tsconfig.json files
log_status "Checking for duplicate tsconfig.json files..." "INFO"
tsconfig_files=$(find "/c/Users/david/.claude" -name "tsconfig*.json" -type f 2>/dev/null | grep -v node_modules | grep -v compiled-servers || true)
tsconfig_count=$(echo "$tsconfig_files" | grep -v '^$' | wc -l)
if [[ $tsconfig_count -gt 0 ]]; then
    log_status "Found $tsconfig_count TypeScript config files:" "INFO"
    echo "$tsconfig_files" | while read -r file; do
        [[ -z "$file" ]] && continue
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        log_status "  - $file ($size bytes)" "INFO"
    done
fi

# Check for duplicate .env files
log_status "Checking for duplicate .env files..." "INFO"
env_files=$(find "/c/Users/david/.claude" -name ".env*" -type f 2>/dev/null | grep -v node_modules | grep -v compiled-servers || true)
env_count=$(echo "$env_files" | grep -v '^$' | wc -l)
if [[ $env_count -gt 0 ]]; then
    log_status "Found $env_count environment files:" "INFO"
    echo "$env_files" | while read -r file; do
        [[ -z "$file" ]] && continue
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        log_status "  - $file ($size bytes)" "INFO"
    done
fi

# Check for duplicate .eslintrc/.prettierrc files
log_status "Checking for duplicate linting/formatting configs..." "INFO"
lint_files=$(find "/c/Users/david/.claude" \( -name ".eslintrc*" -o -name ".prettierrc*" -o -name "biome.json" -o -name ".biome.json" \) -type f 2>/dev/null | grep -v node_modules | grep -v compiled-servers || true)
lint_count=$(echo "$lint_files" | grep -v '^$' | wc -l)
if [[ $lint_count -gt 0 ]]; then
    log_status "Found $lint_count linting/formatting config files:" "INFO"
    echo "$lint_files" | while read -r file; do
        [[ -z "$file" ]] && continue
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        log_status "  - $file ($size bytes)" "INFO"
    done
fi

# Check for duplicate Docker files
log_status "Checking for duplicate Docker files..." "INFO"
docker_files=$(find "/c/Users/david/.claude" \( -name "Dockerfile*" -o -name "docker-compose*" \) -type f 2>/dev/null | grep -v node_modules | grep -v compiled-servers || true)
docker_count=$(echo "$docker_files" | grep -v '^$' | wc -l)
if [[ $docker_count -gt 0 ]]; then
    log_status "Found $docker_count Docker files:" "INFO"
    echo "$docker_files" | while read -r file; do
        [[ -z "$file" ]] && continue
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        log_status "  - $file ($size bytes)" "INFO"
    done
fi

# Check for duplicate lock files
log_status "Checking for duplicate lock files..." "INFO"
lock_files=$(find "/c/Users/david/.claude" \( -name "package-lock.json" -o -name "yarn.lock" -o -name "pnpm-lock.yaml" -o -name "uv.lock" \) -type f 2>/dev/null | grep -v node_modules | grep -v compiled-servers || true)
lock_count=$(echo "$lock_files" | grep -v '^$' | wc -l)
if [[ $lock_count -gt 0 ]]; then
    log_status "Found $lock_count lock files:" "INFO"
    echo "$lock_files" | while read -r file; do
        [[ -z "$file" ]] && continue
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        log_status "  - $file ($size bytes)" "INFO"
    done
fi

# Summary of build directories that could be cleaned
log_status "=== BUILD DIRECTORY ANALYSIS ===" "INFO"
build_dirs=("/c/Users/david/.claude/compiled-servers" "/c/Users/david/.claude/node_modules" "/c/Users/david/.claude/.venv" "/c/Users/david/.claude/dist")

total_size=0
for dir in "${build_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        size=$(du -sb "$dir" 2>/dev/null | cut -f1 || echo "0")
        size_human=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "Unknown")
        log_status "Build directory: $dir = $size_human" "INFO"
        total_size=$((total_size + size))
    fi
done

total_size_gb=$(echo "scale=2; $total_size / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "Unknown")
log_status "Total build directory size: ${total_size_gb}GB" "INFO"

log_status "Duplicate pattern analysis completed" "SUCCESS"
