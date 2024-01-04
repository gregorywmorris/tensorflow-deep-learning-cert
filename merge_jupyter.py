import json
import copy

def merge_two_jupyter_notebooks(first, second):
    """
    Merge Jupyter Notebooks
    See also nbformat
    """

    def read_ipynb(notebook_path):
        """read IPYNB files"""
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def write_ipynb(notebook, notebook_path):
        """export the notebook"""
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f)

    first_notebook = read_ipynb(first)
    second_notebook = read_ipynb(second)

    final_notebook = copy.deepcopy(first_notebook)

    final_notebook['cells'] = first_notebook['cells'] + second_notebook['cells']

    return write_ipynb(final_notebook, 'final_notebook.ipynb')

# merge_two_jupyter_notebooks('00_tensorflow_fundamentals.ipynb',
#                              '01_neural_network_regression_in_tensorflow.ipynb')
