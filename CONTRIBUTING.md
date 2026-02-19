# Contributing to HEXRD

Thank you for your interest in contributing! Whether you want to report a bug, request a feature, or submit a pull request, your contributions are valuable to us.

## How to Contribute

HEXRD is a community-driven project, and we welcome contributions from everyone. To begin a software contribution, follow these steps:

1. [Fork the repository](https://github.com/HEXRD/hexrd/fork)
2. Create a new branch on this fork for your changes
3. Make your changes with clear commit messages

Once you have made your changes, your code will need to meet the conditions below to be approved for integration into the core HEXRD library. At any point in this process, any of the HEXRD Core Developers would be happy to advise and assist in making sure your contribution is ready for integration - please reach out to them.

- [ ] New code is not duplicate with the existing HEXRD codebase - if a HEXRD function already implements a solution, do not re-implement it
- [ ] New code takes advantage of well-used implementations of algorithms in Python libraries where possible
- [ ] To the best of our ability, algorithms are implemented in a way that is efficient from both a compute time and memory use perspective
- [ ] Variable and function names are intuitive or easy to understand within the context of the code
- [ ] Variables and functions are fully type-hinted as granularly as feasible
- [ ] Functions are generally no larger than 50 lines, and at most 80 lines
- [ ] All `TODO` and `FIXME` comments (and related) are resolved
- [ ] Unit tests are written in [pytest](https://docs.pytest.org/en/stable/) and fully cover the lines of added functions - please integrate this into the existing [test suite](https://github.com/HEXRD/hexrd/tree/master/tests)
- [ ] New code uses [`logging`](https://docs.python.org/3/library/logging.html) with appropriate logging levels, instead of `print` statements
- [ ] All functions, methods, and modules are written with docstrings following the [Google styleguide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
- [ ] If HEXRD users will use your new code, descriptive end-user documentation is provided, with examples

Again, the Core Developers are more than happy to help facilitate your code's transition into HEXRD. However, if you would like to improve the readability of your code, the [`pylint`](https://pylint.readthedocs.io/en/stable/) and [`mypy`](https://mypy.readthedocs.io/en/stable/) tools are great to leverage. They will programmatically identify places in your codebase which do not adhere to standards, but should not be taken as gospel truth. When in doubt, discuss your issue with the Core Developers team.

## Pull Request Guidelines

- Ensure your code passes all tests
- Follow the coding style guidelines outlined by [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
- Add or update documentation as needed

## Questions

For questions, check out the conversation threads in the [Discussion](https://github.com/HEXRD/hexrd/discussions) page. If your question is not addressed here, feel free to start a new discussion thread. We welcome all questions and will do our best to address them in a timely manner.

## Reporting Issues

If you have identified a bug or have a feature request, please open an [Issue](https://github.com/HEXRD/hexrd/issues). When reporting an issue, please:

- Search for existing issues before opening a new one
- If reporting a bug, provide detailed information and steps to reproduce it
- If requesting a feature, explain the use case and benefits of the feature

