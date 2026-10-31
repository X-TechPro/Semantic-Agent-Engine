#!/bin/bash
# SemanticMemory Git Commit & Push Script (Linux)

# check if .git folder exists
if [ ! -d ".git" ]; then
    echo "Git repo not initialized. Initializing..."
    git init
    git remote add origin https://github.com/X-TechPro/SemanticMemory.git
fi

# make sure we’re on main
git checkout -q -B main

# add all changes
git add .

# refuse to upload blobs GitHub will reject (>100 MB)
big=$(git diff --cached --name-only --diff-filter=AM | while read -r f; do
    [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 104857600 ] && echo "  $f ($(du -h "$f" | cut -f1))"
done)
if [ -n "$big" ]; then
    echo "ERROR: staged files exceed GitHub's 100 MB limit:"
    echo "$big"
    echo "Add them to .gitignore instead."
    exit 1
fi

# prompt for commit message
read -p "Enter commit message: " msg

# commit
git commit -m "$msg"

# push to origin
git push -u origin main
