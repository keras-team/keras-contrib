# Keras-contrib Documentation

The source for Keras-contrib documentation is in this directory under `sources/`. 
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- install pydoc-markdown: `pip install pydoc-markdown`
- `cd` to the `contrib_docs/` folder and run:
    - `pydocmd serve`    # Starts a local webserver:  [localhost:8000](localhost:8000)
    - `pydocmd build`    # Builds a static site in "site" directory
