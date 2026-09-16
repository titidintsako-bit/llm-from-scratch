#!/bin/bash
# Setup GitHub Repository for LLM From Scratch
# Run this script from the project root directory

echo "🚀 Setting up GitHub repository..."
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo ""
    echo "Install it with:"
    echo "  macOS: brew install gh"
    echo "  Linux: sudo apt install gh"
    echo "  Or visit: https://cli.github.com/"
    echo ""
    echo "Then run: gh auth login"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI."
    echo ""
    echo "Run: gh auth login"
    exit 1
fi

# Initialize git repo
echo "📁 Initializing git repository..."
cd scratchpad
git init
git add -A
git commit -m "Initial commit: LLM from scratch workshop"

# Create GitHub repo
echo ""
echo "🌐 Creating GitHub repository..."
read -p "Enter repository name (e.g., llm-from-scratch): " repo_name
gh repo create "$repo_name" --public --source=. --push

echo ""
echo "✅ Done! Repository created at:"
gh repo view --web

echo ""
echo "📋 Next steps:"
echo "1. Open the repository in your browser"
echo "2. Copy the contents of colab_train.py"
echo "3. Paste into a Google Colab notebook"
echo "4. Set GPU: Runtime → Change runtime type → GPU (T4)"
echo "5. Run the cell!"
