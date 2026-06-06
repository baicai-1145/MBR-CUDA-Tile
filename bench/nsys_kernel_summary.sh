#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 trace.sqlite [limit]" >&2
    exit 2
fi

db="$1"
limit="${2:-40}"

if [[ ! -f "$db" ]]; then
    echo "nsys sqlite file not found: $db" >&2
    exit 1
fi

echo "[nsys] kernel launches"
sqlite3 -header -column "$db" \
    "select count(*) as kernel_launches from CUPTI_ACTIVITY_KIND_KERNEL;"

echo
echo "[nsys] top kernels by total time"
sqlite3 -header -column "$db" "
select
  sid.value as kernel,
  count(*) as calls,
  round(sum(k.end-k.start)/1e6, 3) as total_ms,
  round(avg(k.end-k.start)/1e3, 3) as avg_us
from CUPTI_ACTIVITY_KIND_KERNEL k
join StringIds sid on sid.id = k.demangledName
group by sid.value
order by sum(k.end-k.start) desc
limit ${limit};
"

echo
echo "[nsys] dtype-boundary kernels"
sqlite3 -header -column "$db" "
select
  sid.value as kernel,
  count(*) as calls,
  round(sum(k.end-k.start)/1e6, 3) as total_ms,
  round(avg(k.end-k.start)/1e3, 3) as avg_us
from CUPTI_ACTIVITY_KIND_KERNEL k
join StringIds sid on sid.id = k.demangledName
where lower(sid.value) like '%convert%'
   or lower(sid.value) like '%f32_to%'
   or lower(sid.value) like '%to_f32%'
   or lower(sid.value) like '%bf16_to%'
   or lower(sid.value) like '%to_bf16%'
   or lower(sid.value) like '%f16_to%'
   or lower(sid.value) like '%to_f16%'
   or lower(sid.value) like '%half_to%'
   or lower(sid.value) like '%to_half%'
group by sid.value
order by sum(k.end-k.start) desc
limit ${limit};
"
