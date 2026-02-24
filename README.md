# Visualizing Latent Spaces

Completely overhauled the way this works. Next TODO is to move the pipeline code into the library itself (no more calling Python subprocesses!!!!)  

# Setup
First, go into `frontend` and run `npm install`. Then, run `./build.sh`. This will compile the front-end and place it in the `src` directory. Then, add the `latentvis` library to virtual environment however you like (i.e., through pip with `pip install -e .` or with uv: `uv add --editable [path-to-this-directory]`.

Check out `notebook/example.ipynb` for usage.


# Development
Whenever you want to see a change in the UI, run `./build.sh` and re-run the code in the notebook.

updated
