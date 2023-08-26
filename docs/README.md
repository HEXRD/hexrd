HEXRD Documentation
===================

The HEXRD documentation utilizes sphinx along with sphinx-apidoc to
automatically generate documentation for all of the HEXRD packages.

Manually created files may be added as well in either RST or Markdown format.
See the files in `source/users` as an example.

To build the documentation locally, first install the dependencies in the
`requirements.txt` file, then run the `run_sphinx.sh` script. The HTML files
should be automatically generated and placed inside `build/html`. Open up
the `build/html/index.html` file in a web browser to view the generated
content.
