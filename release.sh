#!/usr/bin/env bash
# Build and stage a release. Used by .github/workflows/release.yml.
#
# Usage:
#   bash release.sh build                    # build the wheel into dist/
#   bash release.sh update-manifest vX.Y.Z   # point pixi/pixi.toml at the
#                                            # released wheel URL and re-lock
#
# The release flow is two-phase because `pixi lock` must download the wheel
# from the URL the manifest names: the workflow first creates the GitHub
# release with the wheel attached, then runs update-manifest, then uploads
# pixi/pixi.toml + pixi/pixi.lock to the same release.
#
# pixi/pixi.lock is committed and re-locked in place, so a release only
# changes the palomero entry; other dependencies keep their pins. Run
# `pixi update --manifest-path pixi/pixi.toml` when you deliberately want
# to refresh them.
set -euo pipefail

cd "$(dirname "$0")"

REPO_URL=https://github.com/Yu-AnChen/palomero

cmd_build() {
    pixi exec --spec python=3.10 --spec python-build pyproject-build --wheel --outdir dist
}

cmd_update_manifest() {
    local tag=$1
    local wheel wheel_url
    wheel=$(basename "$(ls -t dist/palomero-*.whl | head -1)")
    wheel_url="$REPO_URL/releases/download/$tag/$wheel"

    python3 - "$wheel_url" <<'EOF'
import pathlib, re, sys
url = sys.argv[1]
p = pathlib.Path("pixi/pixi.toml")
text = p.read_text()
new = f'palomero = {{ url = "{url}", extras = ["webapp"] }}'
text, n = re.subn(r"^palomero = .*$", new, text, flags=re.M)
if n != 1:
    sys.exit(f"expected exactly one palomero entry in pixi/pixi.toml, found {n}")
p.write_text(text)
print(f"pixi/pixi.toml -> {url}")
EOF

    pixi lock --manifest-path pixi/pixi.toml
}

case "${1:-}" in
    build) cmd_build ;;
    update-manifest) cmd_update_manifest "${2:?usage: release.sh update-manifest vX.Y.Z}" ;;
    *) echo "usage: release.sh {build|update-manifest vX.Y.Z}" >&2; exit 1 ;;
esac
