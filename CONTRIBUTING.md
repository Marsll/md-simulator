# Contributing

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

### Pull Requests

* Do not include issue numbers in the PR title
* Document new code based on the [Documentation Styleguide](#documentation-styleguide)
* End all files with a newline

## Styleguides

### Git Commit Messages

Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Consider starting the commit message with an applicable emoji:
    * :art: `:art:` when improving the format/structure of the code
    * :racehorse: `:racehorse:` when improving performance
    * :non-potable_water: `:non-potable_water:` when plugging memory leaks
    * :memo: `:memo:` when writing docs
    * :bug: `:bug:` when fixing a bug
    * :fire: `:fire:` when removing code or files
    * :white_check_mark: `:white_check_mark:` when adding tests
    * :shirt: `:shirt:` when removing linter warnings

### Python style conventions

We follow the standard Python style conventions described here: [PEP 8](https://www.python.org/dev/peps/pep-0008/)

### Documentation Styleguide

Docstrings should be provided satisfying [PEP 257](https://www.python.org/dev/peps/pep-0257/).

#### Example
##### One-liners

```
def function(a, b):
    """Do X and return a list."""
```

##### Multi-liners
```
def complex(real=0.0, imag=0.0):
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    if imag == 0.0 and real == 0.0:
        return complex_zero
    ...
```
