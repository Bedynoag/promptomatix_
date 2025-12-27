# Guide to Push Repository to GitHub

## Step-by-Step Instructions

### Prerequisites
1. Make sure you have a GitHub account
2. Install Git if not already installed: https://git-scm.com/downloads
3. Configure Git with your name and email (if not done already):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### Step 1: Initialize Git Repository (if not already done)
Navigate to the `promptomatix` directory and run:
```bash
cd promptomatix
git init
```

### Step 2: Create a GitHub Repository
1. Go to https://github.com/new
2. Create a new repository (e.g., name it `promptomatix`)
3. **DO NOT** initialize with README, .gitignore, or license (since you already have these)
4. Copy the repository URL (e.g., `https://github.com/yourusername/promptomatix.git`)

### Step 3: Add All Files and Make Initial Commit
```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed (optional but recommended)
git status

# Make your first commit
git commit -m "Initial commit: Promptomatix project"
```

### Step 4: Connect to GitHub Remote
```bash
# Add GitHub repository as remote (replace with your actual URL)
git remote add origin https://github.com/yourusername/promptomatix.git

# Verify remote was added
git remote -v
```

### Step 5: Push to GitHub
```bash
# Push to main branch (or master if that's your default)
git branch -M main
git push -u origin main
```

If you're using SSH instead of HTTPS:
```bash
git remote add origin git@github.com:yourusername/promptomatix.git
git push -u origin main
```

### Step 6: Authentication
- **HTTPS**: You'll be prompted for username and password/token
  - For password, use a Personal Access Token (not your GitHub password)
  - Create token at: https://github.com/settings/tokens
- **SSH**: Make sure your SSH key is set up with GitHub

## Troubleshooting

### If you get authentication errors:
1. Use Personal Access Token instead of password for HTTPS
2. Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### If you need to update the remote URL:
```bash
git remote set-url origin https://github.com/yourusername/promptomatix.git
```

### To check what files will be pushed:
```bash
git status
git ls-files
```

## Files That Will NOT Be Pushed (Already in .gitignore)
- Virtual environments (`promptomatix_env/`, `.venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Build files (`build/`, `dist/`, `*.egg-info/`)
- Log files (`logs/`, `*.log`)
- Session files (`sessions/`)
- Environment files (`.env`)
- IDE files (`.vscode/`, `.idea/`)
- Third-party libraries (`libs/`)

## Next Steps After Pushing
1. Add a description to your GitHub repository
2. Consider adding topics/tags
3. Update README.md if needed
4. Set up branch protection rules if working with a team

