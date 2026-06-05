#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: $0 binary [function-regex] [limit]" >&2
    exit 2
fi

binary="$1"
filter="${2:-}"
limit="${3:-60}"

if [[ ! -x "$binary" ]]; then
    echo "binary not found or not executable: $binary" >&2
    exit 1
fi

tmp_sass="$(mktemp)"
tmp_res="$(mktemp)"
tmp_rows="$(mktemp)"
trap 'rm -f "$tmp_sass" "$tmp_res" "$tmp_rows"' EXIT

cuobjdump --dump-sass "$binary" | c++filt >"$tmp_sass"
cuobjdump --dump-resource-usage "$binary" | c++filt >"$tmp_res"

echo "[cuobjdump] resource usage"
awk -v filter="$filter" '
function keep() {
    if (func == "") return 0
    if (filter != "" && func !~ filter) return 0
    if (filter == "" && reg < 128 && stack == 0 && shared < 32768) return 0
    return 1
}
/^[[:space:]]*Function[[:space:]]+/ {
    func = $0
    sub(/^[[:space:]]*Function[[:space:]]+/, "", func)
    sub(/:$/, "", func)
    next
}
/^[[:space:]]*REG:/ {
    reg = stack = shared = local = 0
    for (i = 1; i <= NF; ++i) {
        split($i, kv, ":")
        if (kv[1] == "REG") reg = kv[2] + 0
        else if (kv[1] == "STACK") stack = kv[2] + 0
        else if (kv[1] == "SHARED") shared = kv[2] + 0
        else if (kv[1] == "LOCAL") local = kv[2] + 0
    }
    if (keep()) {
        printf "%d\t%d\t%d\t%d\t%s\n", reg, stack, shared, local, func
    }
}
' "$tmp_res" >"$tmp_rows"

printf "REG\tSTACK\tSHARED\tLOCAL\tfunction\n"
sort -t $'\t' -k3,3nr -k1,1nr "$tmp_rows" | head -n "$limit"

echo
echo "[cuobjdump] SASS op counts"
awk -v filter="$filter" '
function reset_counts() {
    instr = hmma = ldsm = ldgsts = ldg = sts = stg = f2fp = f2f = ffma = fadd = fmul = mufu = bar = depbar = imad = iadd3 = prmt = lop3 = shf = 0
}
function keep() {
    if (func == "") return 0
    if (filter != "" && func !~ filter) return 0
    if (filter == "" && hmma == 0) return 0
    return 1
}
function flush() {
    if (!keep()) return
    printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n", \
        instr, hmma, ldsm, ldgsts, ldg, sts, stg, f2fp, f2f, ffma, fadd, fmul, mufu, bar, depbar, imad, iadd3, prmt, lop3, shf, func
}
BEGIN {
    reset_counts()
}
/^[[:space:]]*Function[[:space:]]*:/ {
    flush()
    func = $0
    sub(/^[[:space:]]*Function[[:space:]]*:[[:space:]]*/, "", func)
    reset_counts()
    next
}
{
    line = $0
    if (line !~ /^[[:space:]]*\/\*[0-9a-fA-F]+\*\//) next
    sub(/^[[:space:]]*\/\*[0-9a-fA-F]+\*\/[[:space:]]*/, "", line)
    if (line !~ /^[A-Z@]/) next
    split(line, parts, /[[:space:];]+/)
    op = parts[1]
    instr++
    if (op ~ /^HMMA/) hmma++
    else if (op ~ /^LDSM/) ldsm++
    else if (op ~ /^LDGSTS/) ldgsts++
    else if (op ~ /^LDG/) ldg++
    else if (op ~ /^STS/) sts++
    else if (op ~ /^STG/) stg++
    else if (op ~ /^F2FP/) f2fp++
    else if (op ~ /^F2F/) f2f++
    else if (op ~ /^FFMA/) ffma++
    else if (op ~ /^FADD/) fadd++
    else if (op ~ /^FMUL/) fmul++
    else if (op ~ /^MUFU/) mufu++
    else if (op ~ /^BAR/) bar++
    else if (op ~ /^DEPBAR/) depbar++
    else if (op ~ /^IMAD/) imad++
    else if (op ~ /^IADD3/) iadd3++
    else if (op ~ /^PRMT/) prmt++
    else if (op ~ /^LOP3/) lop3++
    else if (op ~ /^SHF/) shf++
}
END {
    flush()
}
' "$tmp_sass" >"$tmp_rows"

printf "instr\tHMMA\tLDSM\tLDGSTS\tLDG\tSTS\tSTG\tF2FP\tF2F\tFFMA\tFADD\tFMUL\tMUFU\tBAR\tDEPBAR\tIMAD\tIADD3\tPRMT\tLOP3\tSHF\tfunction\n"
sort -t $'\t' -k2,2nr -k1,1nr "$tmp_rows" | head -n "$limit"
