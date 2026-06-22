#!/bin/bash
# Pull solver outputs from Euler into cavity-benchmark/out/.
# Run from the cavity-benchmark root. Missing/failed sources are warned about
# and skipped, so one absent run does not abort the whole pull.
set -uo pipefail
REMOTE="euler:~/semproj"
rc=0

pull() {  # pull <remote-subpath> <local-dest>
  if rsync -a --mkpath --exclude=work/ --exclude=logs/ "$REMOTE/$1/" "$2/"; then
    echo "ok   $1"
  else
    echo "SKIP $1 (missing or failed)" >&2
    rc=1
  fi
}

for solver in bem epgp; do
  for mode in grid ref ksweep; do
    for geom in ellipse sphere; do
      pull "cavity-$solver/out/$mode/$geom" "out/$solver/$mode/$geom"
    done
  done
done
for geom in ellipse sphere; do
  pull "cavity-epgp/out/noise/$geom" "out/epgp/noise/$geom"
done

exit $rc
