"""View locally stored Trackio runs on loopback, without sharing or syncing."""

import argparse

from local_runtime import configure_local_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default="workspaces/local-agent")
    parser.add_argument("--project", default="training-agents-sft")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    configure_local_runtime(args.runtime_dir)
    import trackio

    trackio.show(
        project=args.project,
        server_port=args.port,
        host="127.0.0.1",
        share=False,
        open_browser=False,
        block_thread=True,
    )


if __name__ == "__main__":
    main()
