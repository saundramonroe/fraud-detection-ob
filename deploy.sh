#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# deploy.sh - Deploy Fraud Detection Dashboard to Outerbounds
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Prerequisites:
#   - outerbounds CLI installed and authenticated
#   - Run from the fraud-dash-outerbounds project root
# ============================================================================

APP_NAME="fraud-dashboard"
PORT=8501
PYTHON_VERSION="3.11"

echo "============================================"
echo " Fraud Dashboard - Outerbounds Deploy"
echo "============================================"
echo ""

# ----------------------------------------------------------------------------
# 1. Verify we're in the right directory
# ----------------------------------------------------------------------------

if [[ ! -f "app.py" ]]; then
    echo "ERROR: app.py not found in current directory."
    echo "       Run this script from the project root."
    exit 1
fi

if [[ ! -f "src/config.py" ]]; then
    echo "ERROR: src/config.py not found. The dashboard imports from it."
    exit 1
fi

echo "Project directory: $(pwd)"
echo ""

# ----------------------------------------------------------------------------
# 2. Generate requirements.txt from actual imports
#
#    The dashboard only needs these five packages at runtime.
#    Everything else (scikit-learn, xgboost, shap, etc.) lives on the
#    API server side -- the dashboard calls models over HTTP, it doesn't
#    load them locally.
# ----------------------------------------------------------------------------

cat > requirements.txt <<'EOF'
streamlit
pandas
numpy
plotly
requests
EOF

echo "Created requirements.txt:"
cat requirements.txt
echo ""

# ----------------------------------------------------------------------------
# 3. Deploy
#
#    --package-suffixes py  includes app.py and src/*.py
#    The bash entrypoint works around a known Outerbounds bug where the
#    code packager strips dots from filenames (app.py -> apppy).
# ----------------------------------------------------------------------------

echo "Deploying ${APP_NAME} to Outerbounds..."
echo ""

outerbounds app deploy \
    --name "${APP_NAME}" \
    --app-type web \
    --port "${PORT}" \
    --package-src-path . \
    --package-suffixes py \
    --cpu 2 \
    --memory 4096 \
    --min-replicas 1 \
    --max-replicas 2 \
    --python "${PYTHON_VERSION}" \
    --dep-from-requirements requirements.txt \
    --public-access \
    --description "Fraud Detection Dashboard - Anaconda AI Catalyst" \
    -- bash -c "cd /root/code-package && \
        for f in *py; do mv \$f \${f%py}.py 2>/dev/null; done && \
        for f in src/*py; do mv \$f \${f%py}.py 2>/dev/null; done && \
        streamlit run app.py --server.port ${PORT} --server.headless true"

echo ""
echo "============================================"
echo " Deploy complete"
echo "============================================"
