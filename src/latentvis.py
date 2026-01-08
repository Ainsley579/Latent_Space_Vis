from notebookjs import execute_js
import os

this_dir = os.path.dirname(os.path.realpath(__file__))


# with open(f"{this_dir}/style.css", "r") as f:
#    css_style = f.read()


def showSankey(data, imgFn=lambda x: ""):
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


def showScatter(data, imgFn=lambda x: "test from python"):
    with open(f"{this_dir}/bundle.js", "r") as f:
        code_bundle = f.read()

    images = {}
    for x in data:
        dId = x["id"]
        img = images.get(dId, None)
        if img is None:
            images[dId] = imgFn(dId)

    execute_js(
        library_list=code_bundle,
        main_function="latent_space.Scatterplot",
        data_dict={"data": data, "images": images},
        callbacks={},
        css_list=[],
    )
