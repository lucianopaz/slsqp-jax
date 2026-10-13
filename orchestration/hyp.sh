#!/usr/bin/env bash
# Worktree + branch per hypothesis. Only non-destructive git commands are used.
#
#   hyp.sh new <name>            worktree .worktrees/<name> on new branch hyp/<name> from $BASE
#   hyp.sh base [ref]            detached worktree .worktrees/base-<sha7> at ref (default: $BASE tip)
#   hyp.sh check <name>          verify that python imports slsqp_jax from the worktree
#   hyp.sh run <name> -- <cmd>   run <cmd> in the worktree through `uv run --no-sync`
#   hyp.sh scope <name>          fail if the branch changes files outside the allowed scope
#   hyp.sh merge <name>          merge hyp/<name> into $BASE (--no-ff) from the main checkout
#   hyp.sh drop <name>           remove the worktree (non-forced), keep the branch
#   hyp.sh list                  worktrees and hyp/* branches
#
# Env: SQPDAX_BASE (default local_bench_cleanup), SQPDAX_REQUIRED (space separated paths that
# must exist in $BASE; default lists the tooling files).
set -euo pipefail

die() { echo "hyp.sh: $*" >&2; exit 1; }

COMMON=$(git rev-parse --git-common-dir)
MAIN=$(cd "$COMMON/.." && pwd)
BASE=${SQPDAX_BASE:-local_bench_cleanup}
WTROOT=$MAIN/.worktrees
# Allowed paths (regex, git pathspec relative): sqpdax code + its tests + bookkeeping. Report branches
# (hyp/report-*) may additionally edit the report script.
ALLOWED=${SQPDAX_ALLOWED:-'^(slsqp_jax/sqpdax/|tests/sqpdax/|orchestration/(ledger\.jsonl|queue\.md|CONTEXT\.md|hypotheses/))'}
REPORT_ALLOWED='^scripts/diagnose_benchmark\.py$'
REQUIRED=${SQPDAX_REQUIRED:-"scripts/diagnose_benchmark.py benchmarks/validate.py tests/benchmarks/test_validate.py orchestration/PROTOCOL.md orchestration/hyp.sh"}

cmd=${1:-}; shift || true

ensure_exclude() {
  local ex="$COMMON/info/exclude"
  mkdir -p "$(dirname "$ex")"
  touch "$ex"
  for pat in ".worktrees/" "orchestration/.lock" "diagnostics/"; do
    grep -qxF "$pat" "$ex" || echo "$pat" >> "$ex"
  done
}

wt_path() { echo "$WTROOT/$1"; }

case "$cmd" in
  new)
    name=${1:?usage: hyp.sh new <name>}
    [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "name must match [A-Za-z0-9._-]+"
    git -C "$MAIN" rev-parse --verify -q "refs/heads/$BASE" >/dev/null || die "base branch $BASE not found"
    missing=()
    for f in $REQUIRED; do
      git -C "$MAIN" cat-file -e "$BASE:$f" 2>/dev/null || missing+=("$f")
    done
    if ((${#missing[@]})); then
      die "not tracked on $BASE (commit them first): ${missing[*]}"
    fi
    git -C "$MAIN" rev-parse --verify -q "refs/heads/hyp/$name" >/dev/null && die "branch hyp/$name exists"
    ensure_exclude
    wt=$(wt_path "$name")
    [[ -e $wt ]] && die "$wt exists"
    git -C "$MAIN" worktree add -b "hyp/$name" "$wt" "$BASE"
    [[ -e $MAIN/.venv && ! -e $wt/.venv ]] && ln -s "$MAIN/.venv" "$wt/.venv"
    echo "worktree: $wt"
    echo "branch:   hyp/$name (from $(git -C "$MAIN" rev-parse --short "$BASE"))"
    ;;
  base)
    ref=${1:-$BASE}
    sha=$(git -C "$MAIN" rev-parse --short=7 "$ref")
    ensure_exclude
    wt=$(wt_path "base-$sha")
    if [[ ! -e $wt ]]; then
      git -C "$MAIN" worktree add --detach "$wt" "$ref"
      [[ -e $MAIN/.venv && ! -e $wt/.venv ]] && ln -s "$MAIN/.venv" "$wt/.venv"
    fi
    echo "worktree: $wt"
    ;;
  check)
    name=${1:?usage: hyp.sh check <name>}
    wt=$(wt_path "$name")
    [[ -d $wt ]] || die "no worktree $wt"
    out=$(cd "$wt" && PYTHONPATH="$wt" uv run --no-sync --project "$MAIN" python -c 'import slsqp_jax,sys; print(slsqp_jax.__file__)')
    echo "$out"
    case "$out" in "$wt"/*) echo "ok: importing from the worktree";; *) die "slsqp_jax is imported from outside $wt";; esac
    ;;
  run)
    name=${1:?usage: hyp.sh run <name> -- <cmd...>}; shift
    [[ ${1:-} == -- ]] && shift
    (($#)) || die "no command"
    wt=$(wt_path "$name")
    [[ -d $wt ]] || die "no worktree $wt"
    cd "$wt"
    PYTHONPATH="$wt" exec uv run --no-sync --project "$MAIN" "$@"
    ;;
  scope)
    name=${1:?usage: hyp.sh scope <name>}
    git -C "$MAIN" rev-parse --verify -q "refs/heads/hyp/$name" >/dev/null || die "no branch hyp/$name"
    bad=$(git -C "$MAIN" diff --name-only "$BASE...hyp/$name" | grep -Ev "$ALLOWED" || true)
    if [[ $name == report-* ]]; then bad=$(echo "$bad" | grep -Ev "$REPORT_ALLOWED" || true); fi
    if [[ -n $bad ]]; then echo "outside scope:" >&2; echo "$bad" >&2; exit 1; fi
    echo "scope ok"
    ;;
  merge)
    name=${1:?usage: hyp.sh merge <name>}
    [[ $(git -C "$MAIN" branch --show-current) == "$BASE" ]] || die "main checkout is not on $BASE"
    # bookkeeping files are written in the main checkout during a session; they never conflict with a hypothesis branch
    [[ -z $(git -C "$MAIN" status --porcelain --untracked-files=no -- . ':!orchestration/ledger.jsonl' ':!orchestration/queue.md' ':!orchestration/CONTEXT.md') ]] || die "main checkout has tracked changes; commit or move them first"
    git -C "$MAIN" rev-parse --verify -q "refs/heads/hyp/$name" >/dev/null || die "no branch hyp/$name"
    "$0" scope "$name" || die "branch hyp/$name changes files outside the allowed scope; not merged"
    wt=$(wt_path "$name")
    if [[ -d $wt && -n $(git -C "$wt" status --porcelain --untracked-files=no) ]]; then die "worktree $wt has uncommitted tracked changes"; fi
    if ! git -C "$MAIN" merge --no-ff -m "Merge hyp/$name into $BASE" "hyp/$name"; then
      git -C "$MAIN" merge --abort || true
      die "merge conflict; merge aborted, nothing changed. Resolve with the user."
    fi
    git -C "$MAIN" log --oneline -1
    ;;
  drop)
    name=${1:?usage: hyp.sh drop <name>}
    wt=$(wt_path "$name")
    [[ -d $wt ]] || die "no worktree $wt"
    git -C "$MAIN" worktree remove "$wt" || die "worktree has untracked or modified files; inspect $wt (branch is kept either way)"
    echo "removed $wt; branch kept"
    ;;
  list)
    git -C "$MAIN" worktree list
    git -C "$MAIN" for-each-ref --format='%(refname:short) %(objectname:short) %(subject)' refs/heads/hyp/ refs/heads/report/
    ;;
  *)
    sed -n '2,13p' "$0"; exit 2
    ;;
esac
