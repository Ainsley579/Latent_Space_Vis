from notebookjs import execute_js
import os

this_dir = os.path.dirname(os.path.realpath(__file__))


# with open(f"{this_dir}/style.css", "r") as f:
#    css_style = f.read()


def showSankey(data):
    # TODO: do preprocessing here - the user gives you the torch embeddings, you
    # manipulate them to be ready for your vis

    with open(f"{this_dir}/bundle.js", "r") as f:
        code_bundle = f.read()  # we read the code_bundle file each time so that we don't need to reload the library to see changes in JS

    execute_js(
        library_list=code_bundle,
        main_function="latent_space.SankeyDiagram",
        data_dict=data,
        callbacks={},
        css_list=[],
    )


def showScatter(data):
    with open(f"{this_dir}/bundle.js", "r") as f:
        code_bundle = f.read()
    execute_js(
        library_list=code_bundle,
        main_function="latent_space.Scatterplot",
        data_dict=data,
        callbacks={},
        css_list=[],
    )
