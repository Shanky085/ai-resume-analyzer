# Demo Screenshot

Place the app screenshot at this path: `docs/demo.png`

It is referenced in `README.md` as:

```markdown
![AI Resume Analyzer — App Screenshot](docs/demo.png)
```

## How to capture the screenshot

### 1. Start the app
```bash
streamlit run app.py
```

### 2. Open the browser
Navigate to **http://localhost:8501**

### 3. Produce a meaningful state
- Upload a sample PDF resume (you can use any text-based CV)
- Paste a job description into the text area
- Click **🔍 Analyze Resume**
- Wait for the three progress bars, grade badge, and skill pills to appear

### 4. Capture the screenshot

| OS | Method |
|---|---|
| **Windows** | `Win + Shift + S` → drag to select the browser window → paste into Paint → Save As PNG |
| **macOS** | `Cmd + Shift + 4` → drag over the browser window → file saved to Desktop |
| **Linux** | `gnome-screenshot -a` or use `Flameshot` |

### 5. Save and commit
```bash
# Make sure the docs/ directory exists
mkdir -p docs

# Copy / move your captured file here
cp ~/Desktop/screenshot.png docs/demo.png

# Commit
git add docs/demo.png
git commit -m "docs: add app screenshot"
git push
```

Once pushed, the image will render automatically in `README.md` on GitHub.
