import io
import sys
from contextlib import contextmanager

@contextmanager
def capture_output():
    """
    with capture_output() as output:
        print("This will be captured")
        print("This will also be captured")

    # Retrieve and print the captured output
    captured_text = output.getvalue()
    print("Captured:", captured_text)
    """
    # Create a StringIO object
    captured_output = io.StringIO()
    # Save the original sys.stdout
    original_stdout = sys.stdout
    # Redirect sys.stdout to the StringIO object
    sys.stdout = captured_output
    try:
        yield captured_output
    finally:
        # Reset sys.stdout to its original value
        sys.stdout = original_stdout

# Usage example