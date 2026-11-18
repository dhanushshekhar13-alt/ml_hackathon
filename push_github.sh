#!/bin/bash
# Create the public GitHub repo "ml_hackathon" and push. Uses the GitHub CLI if available.
# Usage:  bash push_github.sh
set -e
cd "$(dirname "$0")"
[ -d .git ] || git init
git add .
git commit -m "PathSnap: data pipeline, EDA, 5 models, HMM, API, UI" || echo "(nothing new to commit)"
git branch -M main
if command -v gh >/dev/null 2>&1; then
  gh auth status >/dev/null 2>&1 || gh auth login
  gh repo view ml_hackathon >/dev/null 2>&1 || gh repo create ml_hackathon --public --source=. --remote=origin
  git push -u origin main
else
  echo "GitHub CLI not installed. Create an empty public repo named ml_hackathon on github.com, then run:"
  echo "  git remote add origin https://github.com/<your-username>/ml_hackathon.git && git push -u origin main"
fi
