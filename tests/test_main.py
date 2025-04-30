import io
from contextlib import redirect_stdout

from main import main


def test_main_output():
    """Tests if main function prints the expected output."""
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        main()
    # Get the captured output, removing potential trailing newline for comparison
    output = captured_output.getvalue().strip()
    assert output == "Hello from cosmic-soup!"
