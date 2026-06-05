#!/usr/bin/env python3
import argparse
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass


OP_PREFIXES = (
    "HMMA",
    "LDSM",
    "LDGSTS",
    "LDG",
    "STS",
    "STG",
    "F2FP",
    "F2F",
    "FFMA",
    "FADD",
    "FMUL",
    "MUFU",
    "BAR",
    "DEPBAR",
    "IMAD",
    "IADD3",
    "PRMT",
    "LOP3",
    "SHF",
)

A10G_DENSE_BF16_TFLOPS = 70.0
A10G_SHARED_PER_SM = 102400
A10G_MAX_BLOCKS_PER_SM = 16
FFN12_M = 78048
FFN12_IN = 256
FFN12_HIDDEN = 1024
FFN12_OUT = 256
TIME_ATTN_BH = 480
TIME_ATTN_N = 1301
TIME_ATTN_MAIN_N = 1280
TIME_ATTN_D = 64
FREQ_ATTN_BH = 10408
FREQ_ATTN_N = 60
FREQ_ATTN_D = 64


@dataclass
class Resource:
    reg: int = 0
    stack: int = 0
    shared: int = 0
    local: int = 0


@dataclass
class SassCounts:
    instr: int = 0
    hmma: int = 0
    ldsm: int = 0
    ldgsts: int = 0
    ldg: int = 0
    sts: int = 0
    stg: int = 0
    f2fp: int = 0
    f2f: int = 0
    ffma: int = 0
    fadd: int = 0
    fmul: int = 0
    mufu: int = 0
    bar: int = 0
    depbar: int = 0
    imad: int = 0
    iadd3: int = 0
    prmt: int = 0
    lop3: int = 0
    shf: int = 0


def run_demangled(cmd: list[str]) -> str:
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return subprocess.check_output(["c++filt"], input=raw).decode("utf-8", "replace")
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.output.decode("utf-8", "replace"))
        raise


def normalize_function_name(name: str) -> str:
    name = name.strip()
    if name.startswith("Function :"):
        name = name[len("Function :"):].strip()
    elif name.startswith("Function "):
        name = name[len("Function "):]
    if name.endswith(":"):
        name = name[:-1]
    name = name.replace("(anonymous namespace)", "<unnamed>")
    name = re.sub(r"^(?:void|static void)\s+", "", name)
    name = name.replace("(bool)0", "false")
    name = name.replace("(bool)1", "true")
    name = re.sub(r"\((?:int|long long|unsigned int|bool)\)", "", name)

    depth = 0
    end = len(name)
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "(" and depth == 0:
            end = i
            break
    name = name[:end]
    name = re.sub(r"\s*,\s*", ",", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def parse_resource_usage(text: str) -> dict[str, Resource]:
    resources: dict[str, Resource] = {}
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function "):
            current = normalize_function_name(stripped)
            continue
        if not current or not stripped.startswith("REG:"):
            continue
        fields = dict(re.findall(r"([A-Z0-9\[\]]+):([0-9]+)", stripped))
        resources[current] = Resource(
            reg=int(fields.get("REG", 0)),
            stack=int(fields.get("STACK", 0)),
            shared=int(fields.get("SHARED", 0)),
            local=int(fields.get("LOCAL", 0)),
        )
    return resources


def parse_sass(text: str) -> dict[str, SassCounts]:
    counts: dict[str, SassCounts] = {}
    current = ""
    cur = SassCounts()

    def flush() -> None:
        nonlocal cur
        if current:
            counts[current] = cur
        cur = SassCounts()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Function :"):
            flush()
            current = normalize_function_name(stripped)
            continue
        if not current:
            continue
        match = re.match(r"/\*[0-9a-fA-F]+\*/\s+(.*)", stripped)
        if not match:
            continue
        body = match.group(1).strip()
        body = re.sub(r"^@!?P[0-9]+\s+", "", body)
        if not body or not re.match(r"^[A-Z]", body):
            continue
        op = re.split(r"[\s;]+", body, maxsplit=1)[0]
        cur.instr += 1
        for prefix in OP_PREFIXES:
            if op.startswith(prefix):
                setattr(cur, prefix.lower(), getattr(cur, prefix.lower()) + 1)
                break
    flush()
    return counts


def nsys_top_kernels(db_path: str, limit: int) -> list[tuple[str, int, float, float]]:
    query = """
select
  sid.value as kernel,
  count(*) as calls,
  sum(k.end-k.start)/1e6 as total_ms,
  avg(k.end-k.start)/1e3 as avg_us
from CUPTI_ACTIVITY_KIND_KERNEL k
join StringIds sid on sid.id = k.demangledName
group by sid.value
order by sum(k.end-k.start) desc
limit ?;
"""
    with sqlite3.connect(db_path) as db:
        return [(row[0], int(row[1]), float(row[2]), float(row[3]))
                for row in db.execute(query, (limit,))]


def diagnose(res: Resource | None, sass: SassCounts | None) -> str:
    reasons: list[str] = []
    if sass is None:
        reasons.append("no SASS match")
        return "; ".join(reasons)
    scalar = sass.ffma + sass.fadd + sass.fmul + sass.mufu
    if sass.hmma == 0:
        reasons.append("non-HMMA scalar/memory path")
    elif sass.hmma <= 32:
        reasons.append("low HMMA density")
    if res is not None:
        if res.shared >= 96 * 1024:
            reasons.append("98KB shared residency wall")
        if res.reg >= 240 or res.stack > 0:
            reasons.append("high reg/stack pressure")
    if sass.mufu >= 64:
        reasons.append("transcendental/softmax MUFU")
    if sass.hmma and scalar >= sass.hmma * 2:
        reasons.append("large scalar side work")
    if sass.hmma and sass.bar >= 30:
        reasons.append("barrier-heavy staging")
    if sass.hmma and sass.imad >= 140:
        reasons.append("address/control overhead")
    if sass.stg >= 3:
        reasons.append("multiple global-store/layout writes")
    if not reasons:
        reasons.append("no obvious static limiter")
    return "; ".join(reasons)


def smem_cta_limit(res: Resource | None) -> int | None:
    if res is None or res.shared <= 0:
        return None
    return min(A10G_MAX_BLOCKS_PER_SM, max(1, A10G_SHARED_PER_SM // res.shared))


def format_smem_cta_limit(res: Resource | None) -> str:
    limit = smem_cta_limit(res)
    if limit is None:
        return "   -"
    return f"{limit:4d}"


def short_name(name: str) -> str:
    name = normalize_function_name(name)
    name = name.replace("cudasep::mbr_tile::<unnamed>::", "")
    name = name.replace("cudasep::tensor_tile::<unnamed>::", "tensor::")
    name = name.replace("cudasep::stft_tile::<unnamed>::", "stft::")
    return name


def split_template_args(name: str) -> tuple[str, list[str]] | None:
    name = normalize_function_name(name)
    if not name.endswith(">"):
        return None
    depth = 0
    start = -1
    for i, ch in enumerate(name):
        if ch == "<":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ">":
            depth -= 1
    if start < 0:
        return None

    args_text = name[start + 1:-1]
    args: list[str] = []
    depth = 0
    token_start = 0
    for i, ch in enumerate(args_text):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(args_text[token_start:i].strip())
            token_start = i + 1
    args.append(args_text[token_start:].strip())
    return name[:start], args


def find_static_entry(mapping, kernel: str):
    norm = normalize_function_name(kernel)
    exact = mapping.get(norm)
    if exact is not None:
        return exact

    parsed = split_template_args(norm)
    if parsed is None:
        return None
    base, args = parsed

    matches = []
    for name, value in mapping.items():
        candidate = split_template_args(name)
        if candidate is None:
            continue
        cand_base, cand_args = candidate
        same_base = (
            cand_base == base or
            cand_base.endswith("::" + base) or
            base.endswith("::" + cand_base)
        )
        if same_base and cand_args[:len(args)] == args:
            matches.append((cand_args, value))

    if len(matches) == 1:
        return matches[0][1]

    default_matches = [
        value for cand_args, value in matches
        if all(arg in ("0", "false") for arg in cand_args[len(args):])
    ]
    if len(default_matches) == 1:
        return default_matches[0]
    return None


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def estimate_useful_flops(kernel: str) -> float | None:
    parsed = split_template_args(kernel)
    if parsed is None:
        return None
    base, args = parsed

    def has_suffix(suffix: str) -> bool:
        return base == suffix or base.endswith("::" + suffix)

    if has_suffix("ffn12_fused256_split2_pairh32_cutile_kernel"):
        return (
            2.0 * FFN12_M * FFN12_HIDDEN * FFN12_IN +
            2.0 * FFN12_M * FFN12_OUT * FFN12_HIDDEN
        )

    if has_suffix("time_attention1301_main1280_split_contig_input_kernel"):
        include_keytail = len(args) >= 4 and args[3] == "true"
        effective_k = TIME_ATTN_N if include_keytail else TIME_ATTN_MAIN_N
        return 4.0 * TIME_ATTN_BH * TIME_ATTN_MAIN_N * effective_k * TIME_ATTN_D

    if has_suffix("time_attention1301_split_contig_tail_kernel"):
        tail_rows = TIME_ATTN_N - TIME_ATTN_MAIN_N
        return 4.0 * TIME_ATTN_BH * tail_rows * TIME_ATTN_N * TIME_ATTN_D

    if has_suffix("freq_attention60_cutile_padded_out60_kernel"):
        return 4.0 * FREQ_ATTN_BH * FREQ_ATTN_N * FREQ_ATTN_N * FREQ_ATTN_D

    static_mnk_kernels = (
        "linear_cutile_static_full_bf16_kernel",
        "linear_cutile_static_full_bkn_bf16_kernel",
        "qkv_bkn_split_contig_static_full_kernel",
        "linear_cutile_static_masked_mn_bf16_kernel",
    )
    if any(has_suffix(name) for name in static_mnk_kernels) and len(args) >= 6:
        m = parse_int(args[3])
        n = parse_int(args[4])
        k = parse_int(args[5])
        if m is not None and n is not None and k is not None:
            return 2.0 * m * n * k

    if has_suffix("linear_cutile_static_padded_m_bf16_kernel") and len(args) >= 7:
        m = parse_int(args[4])
        n = parse_int(args[5])
        k = parse_int(args[6])
        if m is not None and n is not None and k is not None:
            return 2.0 * m * n * k

    return None


def format_metric(value: float | None, width: int, precision: int = 1) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:{width}.{precision}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join Nsight kernel time with cuobjdump resource/SASS bottleneck signals.")
    parser.add_argument("trace_sqlite")
    parser.add_argument("binary")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()

    resources = parse_resource_usage(
        run_demangled(["cuobjdump", "--dump-resource-usage", args.binary]))
    sass_counts = parse_sass(
        run_demangled(["cuobjdump", "--dump-sass", args.binary]))
    rows = nsys_top_kernels(args.trace_sqlite, args.limit * 4)
    filt = re.compile(args.filter) if args.filter else None

    print("[a10g-cutile] per-kernel bottleneck report")
    print(f"trace={args.trace_sqlite}")
    print(f"binary={args.binary}")
    print("note=uses existing Nsight trace and current SASS; does not rerun dense roofline")
    print()
    print(
        f"{'rank':>4} {'calls':>5} {'total_ms':>9} {'avg_us':>8} "
        f"{'TF/s':>7} {'roof%':>6} "
        f"{'REG':>4} {'STACK':>5} {'SMEM':>7} {'smCTA':>5} "
        f"{'HMMA':>5} {'MUFU':>5} "
        f"{'scalar':>6} {'BAR':>4} {'STG':>3}  kernel"
    )
    printed = 0
    for rank, (kernel, calls, total_ms, avg_us) in enumerate(rows, start=1):
        norm = normalize_function_name(kernel)
        if filt and not (filt.search(kernel) or filt.search(norm)):
            continue
        res = find_static_entry(resources, norm)
        sass = find_static_entry(sass_counts, norm)
        if sass is None:
            sass = SassCounts()
        scalar = sass.ffma + sass.fadd + sass.fmul + sass.mufu
        useful_flops = estimate_useful_flops(norm)
        useful_tflops = None
        roof_pct = None
        if useful_flops is not None and total_ms > 0.0:
            useful_tflops = (useful_flops * calls) / (total_ms * 1.0e-3) / 1.0e12
            roof_pct = useful_tflops * 100.0 / A10G_DENSE_BF16_TFLOPS
        print(
            f"{rank:4d} {calls:5d} {total_ms:9.3f} {avg_us:8.3f} "
            f"{format_metric(useful_tflops, 7)} {format_metric(roof_pct, 6)} "
            f"{res.reg if res else 0:4d} {res.stack if res else 0:5d} "
            f"{res.shared if res else 0:7d} {format_smem_cta_limit(res)} "
            f"{sass.hmma:5d} {sass.mufu:5d} "
            f"{scalar:6d} {sass.bar:4d} {sass.stg:3d}  {short_name(kernel)}"
        )
        print(f"     cause: {diagnose(res, find_static_entry(sass_counts, norm))}")
        printed += 1
        if printed >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
