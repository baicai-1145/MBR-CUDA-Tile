#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: $0 trace.sqlite [cutoff_ms] [limit]" >&2
    exit 2
fi

db="$1"
cutoff_ms="${2:-100}"
limit="${3:-40}"

if [[ ! -f "$db" ]]; then
    echo "nsys sqlite file not found: $db" >&2
    exit 1
fi

case "$cutoff_ms" in
    ''|*[!0-9]*)
        echo "cutoff_ms must be a non-negative integer" >&2
        exit 2
        ;;
esac

echo "[nsys] dtype/conversion kernel timeline"
sqlite3 -header -column "$db" "
with
  bounds as (
    select min(start) as t0, max(end) as t1
    from CUPTI_ACTIVITY_KIND_KERNEL
  ),
  named as (
    select
      k.start,
      k.end,
      sid.value as kernel
    from CUPTI_ACTIVITY_KIND_KERNEL k
    join StringIds sid on sid.id = k.demangledName
  ),
  filtered as (
    select *
    from named
    where lower(kernel) like '%convert%'
       or lower(kernel) like '%f32_to%'
       or lower(kernel) like '%to_f32%'
       or lower(kernel) like '%bf16_to%'
       or lower(kernel) like '%to_bf16%'
       or lower(kernel) like '%f16_to%'
       or lower(kernel) like '%to_f16%'
       or lower(kernel) like '%half_to%'
       or lower(kernel) like '%to_half%'
  )
select
  kernel,
  count(*) as calls,
  round(sum(end - start) / 1e6, 3) as total_ms,
  round(avg(end - start) / 1e3, 3) as avg_us,
  round((min(start) - (select t0 from bounds)) / 1e6, 3) as first_ms,
  round((max(end) - (select t0 from bounds)) / 1e6, 3) as last_ms
from filtered
group by kernel
order by sum(end - start) desc
limit ${limit};
"

echo
echo "[nsys] dtype/conversion kernels after ${cutoff_ms} ms"
sqlite3 -header -column "$db" "
with
  bounds as (
    select min(start) as t0
    from CUPTI_ACTIVITY_KIND_KERNEL
  ),
  named as (
    select
      k.start,
      k.end,
      sid.value as kernel
    from CUPTI_ACTIVITY_KIND_KERNEL k
    join StringIds sid on sid.id = k.demangledName
  ),
  filtered as (
    select *
    from named
    where start >= (select t0 + ${cutoff_ms} * 1000000 from bounds)
      and (
         lower(kernel) like '%convert%'
      or lower(kernel) like '%f32_to%'
      or lower(kernel) like '%to_f32%'
      or lower(kernel) like '%bf16_to%'
      or lower(kernel) like '%to_bf16%'
      or lower(kernel) like '%f16_to%'
      or lower(kernel) like '%to_f16%'
      or lower(kernel) like '%half_to%'
      or lower(kernel) like '%to_half%'
      )
  )
select
  kernel,
  count(*) as calls,
  round(sum(end - start) / 1e6, 3) as total_ms,
  round(avg(end - start) / 1e3, 3) as avg_us,
  round((min(start) - (select t0 from bounds)) / 1e6, 3) as first_ms,
  round((max(end) - (select t0 from bounds)) / 1e6, 3) as last_ms
from filtered
group by kernel
order by sum(end - start) desc
limit ${limit};
"
