"""Entry point for ``python -m manga``.

Dispatches to the unified CLI in ``manga.cli.main``.
"""

from manga.cli.main import main

if __name__ == "__main__":
    main()
