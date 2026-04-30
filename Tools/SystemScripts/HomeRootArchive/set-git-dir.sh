#!/bin/bash
# This script sets the Git clone directory based on hostname
hostname=$(hostname -s | tr '[:upper:]' '[:lower:]')

# Define machine-specific paths
case "$hostname" in
"work-mbp")
	export GIT_CLONE_DIR="$HOME/Projects"
	;;
"linux-desktop")
	export GIT_CLONE_DIR="/data/git"
	;;
"personal-mac")
	export GIT_CLONE_DIR="$HOME/codedev"
	;;
*)
	# Default fallback location
	export GIT_CLONE_DIR="$HOME/Documents/GitHub"
	;;
esac

# Output the selected directory
echo "Git clone directory set to: $GIT_CLONE_DIR"
