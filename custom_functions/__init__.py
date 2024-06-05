from .ann_functions import *
from . import utils as utils
from . import model_logging as logs
from . import fileinfo as fileinfo
from . import quantize_model as quantize
from . import capture_output as capture

def show_code(function):
	"""
	Uses the inspect module to retrieve the source code for a function.
	Displays the code as Python-syntax Markdown code.

	Note: Python highlighting may not work correctly on some editors.

	Parameters:
	function (callable): The function for which to display the source code.

	Returns:
	None
	"""
	import inspect
	from IPython.display import display, Markdown

	code = inspect.getsource(function)
	md = "```python" +'\n' + code + "\n" + '```' 
	display(Markdown(md))

