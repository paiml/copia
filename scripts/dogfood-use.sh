#!/usr/bin/env bash
# copia dogfood-use - exercise the RELEASE binary on a realistic filesystem tree
# through every sync mode. Invoked by the /dogfood skill with:
#   BIN  = the freshly-built release binary (absolute path)
#   WORK = a scratch directory
# Exits non-zero if the shipped tool misbehaves. All filesystem ops run with
# RELATIVE paths inside the validated scratch dir (no absolute/traversal paths).
set -uo pipefail

BIN="${BIN:?set BIN to the copia release binary}"
WORK="${WORK:?set WORK to a scratch dir}"
[ -d "$WORK" ] || { echo "dogfood-use: WORK is not a directory" >&2; exit 1; }
cd "$WORK" || { echo "dogfood-use: cannot enter WORK" >&2; exit 1; }
export HOME="$PWD/home"
mkdir -p home
fail() { echo "DOGFOOD-USE FAIL: $1" >&2; exit 1; }

# A realistic tree: text, nested dirs, a config, and a binary blob.
mkdir -p dataset/sub/deep
printf 'name = "demo"\nversion = "1"\n' > dataset/app.toml
base64 /dev/urandom | head -c 40000 > dataset/notes.txt
echo "nested content" > dataset/sub/leaf.txt
head -c 200000 /dev/urandom > dataset/sub/deep/blob.bin

# L1: recursive one-way sync - byte-identical, then incremental on re-run.
"$BIN" sync -r dataset/ l1/ >/dev/null 2>&1 || fail "L1 sync errored"
diff -r dataset l1 >/dev/null 2>&1 || fail "L1 sync not byte-identical"
res="$("$BIN" sync -r dataset/ l1/ 2>&1)"
printf '%s' "$res" | grep -qE '0 to transfer|up to date' || fail "L1 re-sync not incremental"
"$BIN" sync -r --exclude '*.bin' dataset/ l1e/ >/dev/null 2>&1 || fail "L1 --exclude errored"
[ ! -e l1e/sub/deep/blob.bin ] || fail "L1 --exclude did not drop *.bin"

# L2: bidirectional sync - propagation both ways.
mkdir -p a b
echo seed > a/f
echo seed > b/f
"$BIN" bisync a b >/dev/null 2>&1 || fail "L2 bisync seed errored"
echo edited-on-a > a/f
"$BIN" bisync a b >/dev/null 2>&1 || fail "L2 bisync propagate errored"
[ "$(cat b/f)" = edited-on-a ] || fail "L2 bisync did not propagate A->B"

# L3: hub - push real data to a local hub, verify byte-identity.
"$BIN" hub-sync dataset hub >/dev/null 2>&1 || fail "L3 hub-sync errored"
diff -r dataset hub --exclude=.copia >/dev/null 2>&1 || fail "L3 hub tree not byte-identical"

echo "dogfood-use OK: L1 sync (+incremental +exclude), L2 bisync propagate, L3 hub round-trip - verified on real data"
