"""Entry point for ``python -m src``.

Dispatches to the unified CLI in ``src.cli.main``.
"""

from src.cli.main import main

if __name__ == "__main__":
    main()
