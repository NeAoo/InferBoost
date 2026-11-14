"""Small adapter around the unmodified official vLLM sweep implementation."""
import signal


def stop(signum, frame):
    raise KeyboardInterrupt('signal ' + str(signum))


def main():
    from vllm.benchmarks.sweep.serve import SweepServeArgs, run_main
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    signal.signal(signal.SIGTERM, stop)
    parser = FlexibleArgumentParser()
    SweepServeArgs.add_cli_args(parser)
    run_main(SweepServeArgs.from_cli_args(parser.parse_args()))


if __name__ == '__main__':
    main()
