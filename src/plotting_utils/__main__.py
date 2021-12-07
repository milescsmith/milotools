"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Plotting Utils."""


if __name__ == "__main__":
    main(prog_name="plotting-utils")  # pragma: no cover
