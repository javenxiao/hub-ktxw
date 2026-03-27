#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
from pathlib import Path


TARGET = "armv7-unknown-linux-gnueabihf"
PROFILE = "release"
BIN = "wireless_status_server"
ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "target" / TARGET / PROFILE / BIN


def require_tool(name: str) -> str:
    tool_path = shutil.which(name)
    if tool_path is None:
        print(f"missing tool: {name}", file=sys.stderr)
        raise SystemExit(1)
    return tool_path


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    strip_tool = os.environ.get(
        "CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_STRIP",
        "arm-linux-gnueabihf-strip",
    )
    upx_tool = os.environ.get("UPX", "upx")

    run_command(["cargo", "build", "--release", "--target", TARGET])

    require_tool(strip_tool)
    require_tool(upx_tool)

    run_command([strip_tool, str(ARTIFACT)])
    run_command([upx_tool, "--best", str(ARTIFACT)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())