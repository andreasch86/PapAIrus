import click


class NoChangesWarning(click.ClickException):
    exit_code = 3

    def show(self, file=None):
        if file is None:
            file = click.get_text_stream("stderr")
        click.secho(f"Warning: {self.format_message()}", fg="yellow", file=file)


class MissingEmbeddingModelError(click.ClickException):
    """Raised when the configured embedding model is unavailable in Ollama."""

    exit_code = 4

    def show(self, file=None):
        if file is None:
            file = click.get_text_stream("stderr")
        click.secho(f"Error: {self.format_message()}", fg="red", file=file)
