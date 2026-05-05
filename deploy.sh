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
#   - Run from the fraud-detection-ob project root
# ============================================================================

APP_NAME="fraud-dashboard"
PORT=8501
PYTHON_VERSION="3.11"
DASHBOARD_FILE="fraud-dash-outerbounds.py"

echo "============================================"
echo " Fraud Dashboard - Outerbounds Deploy"
echo "============================================"
echo ""

# ----------------------------------------------------------------------------
# 1. Verify we're in the right directory
# ----------------------------------------------------------------------------

if [[ ! -f "${DASHBOARD_FILE}" ]]; then
    echo "ERROR: ${DASHBOARD_FILE} not found in current directory."
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
# 2. Create app.py from the dashboard file
#    The deploy command uses app.py as the entry point.
# ----------------------------------------------------------------------------

echo "Copying ${DASHBOARD_FILE} -> app.py"
cp "${DASHBOARD_FILE}" app.py
echo ""

# ----------------------------------------------------------------------------
# 3. Generate requirements.txt
#    scikit-learn is required for model loading and scoring.
# ----------------------------------------------------------------------------

cat > requirements.txt <<'EOF'
streamlit
pandas
numpy
plotly
requests
scikit-learn
EOF

echo "Created requirements.txt:"
cat requirements.txt
echo ""

# ----------------------------------------------------------------------------
# 4. Deploy
#
#    --package-suffixes py,csv,pkl,yml includes code, data, models, config
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
    --package-suffixes py,csv,pkl,yml \
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
