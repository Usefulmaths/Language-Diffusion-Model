#!/usr/bin/env python3
"""Main entry point for diffusion-language-model."""

from src.utils import hello


def main() -> None:
    """Run the main application."""
    result = hello()
    print(result["message"])


if __name__ == "__main__":
    main()
