# Contributing to chemotools

Thanks for your interest in contributing to **`chemotools`** 🎉 
We welcome bug reports, feature ideas, code improvements, and documentation updates. Every contribution helps!

You can also check the [Project Board](https://github.com/users/paucablop/projects/4) to see what’s currently in progress.

---

## Guidelines

`chemotools` is designed for production use, with a focus on **quality and consistency**.  
When contributing, please keep in mind:

- ✅ **Follow the Scikit-Learn API**  
  All transformers must implement the [scikit-learn API](https://scikit-learn.org/stable/developers/develop.html) for interoperability.  

- ✅ **Avoid redundancy**  
  Don’t re-implement functionality already available in other libraries that follow the same API.  

- ✅ **Write tests**  
  Every new function or fix must include unit tests to ensure reliability.  

- ✅ **Maintain quality**  
  Use the provided tooling (formatting, linting, typing, tests) to keep the codebase consistent.  

---

## How to Contribute

1. **Report issues**  
   - For bugs: open an [issue](https://github.com/paucablop/chemotools/issues) with steps to reproduce and sample code/data if possible.  
   - For enhancements: describe the idea, its benefits, and example usage.  

2. **Propose changes**  
   - Open an issue first so we can discuss scope.  
   - Create a new branch for your contribution.  
   - Make your changes, including tests and documentation updates if needed. 
   - Branches should be shor-lived with a well defined scope, following a [truck-based development](https://trunkbaseddevelopment.com/) philosophy.

3. **Check your work**  
   Use the [Taskfile](./Taskfile.yml) for a quick workflow:  

   ```bash
   task install     # install dependencies
   task check       # run formatting, linting, typing, tests
   task coverage    # run tests with coverage
   task build       # build the package
   ```

4. **Open a Pull Request (PR)**

   * Explain what the change does and why.
   * Ensure CI checks pass (formatting, lint, type checks, tests).
   * Be responsive to feedback.


## Code Style

* Code is automatically formatted and linted using [Ruff](https://docs.astral.sh/ruff/).
* Type checking is done with [MyPy](http://mypy-lang.org/).
* Follow general Python best practices: descriptive names, no magic numbers, clear docstrings.



## Testing & Coverage

* Run tests with:

  ```bash
  task test
  ```

* Coverage is tracked with [Codecov](https://codecov.io/). PRs should not reduce coverage.

---

## License

By contributing, you agree that your contributions will be licensed under the project’s [MIT License](LICENSE).


