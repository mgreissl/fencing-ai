"""Single command line entry point: ``python -m src <command> [args...]``.

Every command forwards its remaining arguments unchanged to the underlying
module, run as ``python -m <module>`` from the repository root in a
subprocess. Hydra commands (``train``, ``evaluate``) take ``key=value``
overrides, the others take the argparse options of their script
(``python -m src <command> --help`` shows them).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

# command -> (module, help)
COMMANDS: dict[str, tuple[str, str]] = {
    "harvest": ("scripts.harvest_fencingvision", "collect and verify match URLs"),
    "preprocess": ("scripts.preprocess_pipeline", "download, cut and label clips"),
    "precompute-kinematics": (
        "scripts.precompute_kinematics",
        "cache pose kinematics for clips",
    ),
    "train": ("src.train", "train the model (Hydra overrides)"),
    "evaluate": ("src.evaluate", "evaluate a checkpoint (Hydra overrides)"),
    "bootstrap": ("scripts.bootstrap_eval", "bootstrap CIs and McNemar test"),
    "ablation": ("scripts.run_ablation", "run the ablation grid"),
    "audit-sample": ("scripts.audit_sample", "draw a label audit sample"),
    "audit-report": ("scripts.audit_report", "summarise a label audit"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencing-ai",
        description=__doc__.split("\n")[0],
        epilog="Arguments after the command are passed through unchanged.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (_, help_text) in COMMANDS.items():
        # add_help=False so that --help reaches the underlying module
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def module_exists(module: str) -> bool:
    return (REPO_ROOT / Path(*module.split("."))).with_suffix(".py").exists()


def run_command(command: str, extra_args: Sequence[str]) -> int:
    """Run one command in a subprocess and return its exit code."""
    module, _ = COMMANDS[command]
    if not module_exists(module):
        print(f"fencing-ai: '{command}' is not available ({module} missing)",
              file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", module, *extra_args]
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().parse_args(["--help"])
    command, *extra_args = argv
    if command not in COMMANDS:
        build_parser().error(
            f"invalid command '{command}' (choose from {', '.join(COMMANDS)})"
        )
    return run_command(command, extra_args)


if __name__ == "__main__":
    sys.exit(main())
