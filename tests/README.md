# ✅❌ Good practices for testing with ``pytest`` 🤔

## 1) ✍️ Type hint your tests
While this might seem tedious because tests are not the actual code after all, this helps catching errors early when writing tests, especially those one did not think of before. A good type checker will help here by underlining type mismatches in red.

## 2) 🪛 Test your test utility functions
Some tests require utility functions to be written. Since a utility function is also code that can have bugs, it is important to test it as well.<br>
They can either be tested with a dedicated test

```python
def utility_function():
    return 2.0

def test_utility_function():
    assert utility_function() == 2.0
```

or via a doctest that will be included in ``chemotools``' test suite by ``pytest``.

```python
def utility_function():
    """
    Doctests
    --------
    >>> utility_function()
    2.0
    """

    return 2.0
```

## 3) 🦾🤖 Don't write the same test twice - use parametrization
If you have a test that is repeated with different inputs, use parametrisation to avoid writing the same test twice.<br>
This will make your test suite more readable and maintainable. With the ``pytest.mark.parametrize`` decorator, you can run the same test with different inputs. In the following example, the test will run 5 times with the inputs 1, 4, 9, 16, and 25.

```python
import pytest

@pytest.mark.parametrize("input", [1, 4, 9, 16, 25])
def test_is_square(input: int) -> None:
    assert input ** 0.5 == int(input ** 0.5)
```

In case you want to test multiple input combinations, you can use multiple wrappings of `@pytest.mark.parametrize`. The next test will run 5 x 5 = 25 combinations of inputs.

```python
import pytest

@pytest.mark.parametrize("input_2", [1, 4, 9, 16, 25])
@pytest.mark.parametrize("input_1", [1, 4, 9, 16, 25])
def test_sum_is_positive(
    input_1: int,
    input_2: int
) -> None:
    assert input_1 + input_2 > 0
```

If you need multiple wrappings, but some combinations are not valid, you can use `pytest.skip` to skip the test. The following will run 5 x 5 = 25 tests, but will skip the test when both inputs are 1.

```python
import pytest

@pytest.mark.parametrize("input_2", [1, 4, 9, 16, 25])
@pytest.mark.parametrize("input_1", [1, 4, 9, 16, 25])
def test_sum_is_positive(
    input_1: int,
    input_2: int
) -> None:
    if input_1 == 1 and input_2 == 1:
        pytest.skip("This test is not valid")

    assert input_1 + input_2 > 0
```

Finally, in case your test runs on multiple specific combinations of inputs and expected outputs, you can parametrize the full combination in a ``pytest.mark.parametrize`` decorator.

```python
import pytest


@pytest.mark.parametrize(
    "input_1, input_2, expected",
    [
        (1, 2, 3),
        (2, 3, 5),
        (3, 4, 7),
    ],
)
def test_sum_is_correct(
    input_1: int,
    input_2: int,
    expected: int,
) -> None:
    assert input_1 + input_2 == expected
```

## 4) 💣❌ Test your error handling thoroughly based on error messages
If your function raises an error, you should test that it raises the correct error. You can use the ``pytest.raises`` context manager to check that the function raises the expected error.

```python
import pytest

def divide(a: int, b: int) -> float:
    return a / b

def test_divide_by_zero_raises_error() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
```

Note that it's crucial to put a ``return`` statement at the end of an error test to avoid that everything that comes after the test is also executed.

```python
from typing import List, Union

import numpy as np
import pytest


def function_for_an_array(input: Union[List[int], np.ndarray]) -> np.ndarray:
    if isinstance(input, list):
        raise TypeError("Input must be a numpy array")

    # Do something with the input
    return input

@pytest.mark.parametrize(
    "input",
    [
        [1, 2, 3],
        np.array([1, 2, 3]),
    ],
)
def test_function_for_an_array(
    input: Union[List[int], np.ndarray],
) -> None:
    if isinstance(input, list):
        with pytest.raises(TypeError):
            function_for_an_array(input)

        return  # without this, the following code would still be executed and fail

    result = function_for_an_array(np.array(input))
    assert result is not None

```

However, this is not reliable enough for functions that can raise the same exception type in different contexts. In this case, you can use the ``match`` argument of ``pytest.raises`` to check the error message. For the next function to test, a ``ValueError`` will be encountered for both ``a`` and ``b`` being negative.

```python
import pytest

def sum_non_negative_values(a: int, b: int) -> int:
    if a < 0:
        raise ValueError("a must be non-negative")

    if b < 0:
        raise ValueError("b must be non-negative")

    return a + b
```

Now, the following test will pass but ``b`` being negative is never tested.

```python
@pytest.mark.parametrize(
    "a, b, expected",
    [
        (-1, 1, ValueError()),
        (-1, -1, ValueError()),
    ],
)
def test_sum_non_negative_values_raises_error(
    a: int,
    b: int,
    expected: Exception,
) -> None:
    with pytest.raises(type(expected)):
        sum_non_negative_values(a, b)
```

``b`` being negative is never hit because ``a`` is negative in both tests. Yet, the ``ValueError`` is still properly raised. Such a situation can be avoided by using the ``match`` argument of ``pytest.raises`` to catch and check the error message.

```python
@pytest.mark.parametrize(
    "a, b, expected",
    [
        (-1, 1, ValueError("a must be non-negative")),
        (-1, -1, ValueError("b must be non-negative")),  # this test will fail
    ],
)
def test_sum_non_negative_values_raises_error(
    a: int,
    b: int,
    expected: Exception,
) -> None:
    error_catch_phrase = str(expected)
    with pytest.raises(type(expected), match=error_catch_phrase):
        sum_non_negative_values(a, b)
```

Due to the enhanced test, the test will now fail with the following output:

```bash
        with pytest.raises(ValueError, match=expected):
>           sum_non_negative_values(a, b)
E           AssertionError: Regex pattern did not match.
E            Regex: 'b must be non-negative'
E            Input: 'a must be non-negative'
```

Of course, the same principles apply for warnings that can be caught with ``pytest.warns``.

```python
import pytest

def function_that_warns() -> None:
    import warnings
    warnings.warn("This is a warning", UserWarning)

def test_function_that_warns() -> None:
    with pytest.warns(UserWarning, match="This is a warning"):
        function_that_warns()

    return
```

## 5) 🧪🧫 Test edge cases
Edge cases are the limits of the input space. They are often the source of bugs in code.<br>
Let's say your function starts to misbehave when the input is 0. You should write a test
for that case.

```python
import pytest

def divide(a: int, b: int) -> float:
    return a / b

def test_divide_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
```