# Team development

Clone the repository and follow the setup in README.md. Run all commands from the repository root; training data and checkpoint paths depend on it.

Create a branch for each change, commit the relevant source files, push the branch, and open a pull request:

```bash
git switch -c feature/your-change
python -m pytest -q
node --check deployment/static/app.js
git status --short
git add <specific-files>
git commit -m "Describe the change"
git push -u origin HEAD
```

Replace the example branch and file names with your own. Include what changed, how you tested it, and screenshots for interface changes in your pull request. Have another teammate review before merging.

Keep model weights, research images, camera captures, environment files, and generated training output out of Git. Share the two required model files separately through an agreed team download or GitHub release and record their source/version. Load checkpoints only from a trusted source.

The Python tests use isolated fixtures and do not require trained checkpoints. Camera behavior needs a manual browser test: allow permission, capture a fruit, capture again and check that results replace the previous snapshot, then stop the camera. Check permission-denied behavior too. Node.js is needed only for the JavaScript syntax check.

Predictions are not training labels. Have a person verify labels before adding images to a training set, and keep validation images separate from training images and their augmented copies.
