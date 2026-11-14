# Sourced only by interactive shells; batch commands keep their normal output.
case $- in *i*) ;; *) return ;; esac
if [ -t 1 ] && [ -z "${INFERBOOST_WELCOME_SHOWN:-}" ]; then
    INFERBOOST_WELCOME_SHOWN=1
    PYTHONPATH=/opt/pipeline /opt/venv/bin/python -m pipeline.terminal
fi
