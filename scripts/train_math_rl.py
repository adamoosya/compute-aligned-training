#!/usr/bin/env python3
from cat.experiments.rl import main
if __name__ == "__main__":
    raise SystemExit(main(task_filter="math", evaluate=False))
