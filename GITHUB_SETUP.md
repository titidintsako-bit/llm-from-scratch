# GitHub Setup Instructions

## Option 1: Using GitHub CLI (Recommended)

```bash
# Install GitHub CLI (if not installed)
# macOS
brew install gh

# Linux (Ubuntu/Debian)
sudo apt install gh

# Windows
winget install GitHub.cli

# Login to GitHub
gh auth login

# Navigate to project
cd scratchpad

# Initialize and push
git init
git add -A
git commit -m "Initial commit: LLM from scratch workshop"
gh repo create llm-from-scratch --public --source=. --push
```

## Option 2: Manual Setup

1. **Go to** [github.com/new](https://github.com/new)
2. **Create a new repository**:
   - Name: `llm-from-scratch`
   - Description: "Build a GPT language model from scratch"
   - Public
   - Don't initialize with README (we have our own)
3. **Click "Create repository"**
4. **Follow the "push an existing repository" instructions**:

```bash
cd scratchpad
git init
git add -A
git commit -m "Initial commit: LLM from scratch workshop"
git remote add origin https://github.com/YOUR-USERNAME/llm-from-scratch.git
git branch -M main
git push -u origin main
```

## Option 3: Upload via GitHub Web Interface

1. **Go to** [github.com/new](https://github.com/new)
2. **Create a new repository**:
   - Name: `llm-from-scratch`
   - Public
3. **After creation, click "uploading an existing file"**
4. **Drag and drop all files from the `scratchpad/` folder**
5. **Click "Commit changes"**

## Using from Google Colab

Once your repo is on GitHub, you can use it in Colab:

```python
# In a Colab cell, clone your repo
!git clone https://github.com/YOUR-USERNAME/llm-from-scratch.git
%cd llm-from-scratch/scratchpad

# Install dependencies
!pip install -r requirements.txt

# Run training
!python train.py
```

Or copy-paste the contents of `colab_train.py` directly into a Colab cell.

## Files to Upload

Make sure these files are in your repository:

```
llm-from-scratch/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── colab_train.py           # Complete script for Google Colab
├── scratchpad/
│   ├── model.py             # GPT transformer architecture
│   ├── train.py             # Training loop
│   ├── generate.py          # Text generation
│   ├── attention_deep_dive.py   # Attention explanation
│   └── backprop_deep_dive.py    # Backpropagation explanation
└── data/
    └── shakespeare.txt      # Training dataset
```

## Troubleshooting

**"Permission denied"**
- Make sure you're logged in: `gh auth status`
- Or use HTTPS URL instead of SSH

**"Repository already exists"**
- Use a different name: `gh repo create my-llm-workshop --public`

**"Git not found"**
- Install git: `sudo apt install git` (Linux) or `brew install git` (macOS)
