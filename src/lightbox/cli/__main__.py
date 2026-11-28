"""Allow invoking Lightbox via `python -m lightbox.cli ...` (including Windows `pythonw.exe`)."""

from __future__ import annotations

from lightbox.cli.app import main

if __name__ == "__main__":
    main()
