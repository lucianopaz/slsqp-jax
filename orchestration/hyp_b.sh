#!/usr/bin/env bash
# Stream B: same commands as hyp.sh, but its own integration branch (stream/b), its own checkout
# (../slsqp-jax-streamB) and its own worktree folder, so that the two streams never touch the same files.
#   orchestration/hyp_b.sh new|base|check|run|scope|merge|drop|list ...
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir)
MAIN=$(cd "$COMMON/.." && pwd)
export SQPDAX_BASE=${SQPDAX_BASE:-stream/b}
export SQPDAX_MERGE_DIR=${SQPDAX_MERGE_DIR:-$(dirname "$MAIN")/slsqp-jax-streamB}
export SQPDAX_WTROOT=${SQPDAX_WTROOT:-$SQPDAX_MERGE_DIR/.worktrees}
exec "$MAIN/orchestration/hyp.sh" "$@"
