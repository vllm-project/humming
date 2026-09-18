# Agent Guidelines

## Code Guidelines

### Follow Style Configuration Files

- Follow `.clang-format` for C++ and CUDA formatting.
- Follow the Ruff and mypy settings in `pyproject.toml` for Python formatting, linting, and type checking.
- C++ and CUDA code has no fixed line-length limit. Follow the style of similar files or the code surrounding your changes.

### Split Complex Expressions

When a complex expression becomes long enough to require line wrapping, consider splitting it into multiple statements with clearly named intermediate variables to improve readability.

Nested expressions are acceptable when they fit comfortably within the applicable line-length guidance and remain easy to read. Avoid aggressive splitting; introduce intermediate variables only when they improve readability.

Preserve short-circuit behavior when later checks depend on earlier conditions, have side effects, or are expensive to evaluate.

**BAD**: A complex condition requires line wrapping and obscures the checks being performed.

```python
if (
    inputs.dtype == torch.float16
    and weight.dtype == torch.float16
    and inputs.is_contiguous()
    and weight.is_contiguous()
    and inputs.shape[-1] % tile_size == 0
):
    return run_optimized_kernel(inputs, weight)
```

**GOOD**: Group related checks into clearly named boolean variables to make the condition easier to read.

```python
has_supported_dtype = inputs.dtype == torch.float16 and weight.dtype == torch.float16
has_contiguous_layout = inputs.is_contiguous() and weight.is_contiguous()
is_tile_aligned = inputs.shape[-1] % tile_size == 0

if has_supported_dtype and has_contiguous_layout and is_tile_aligned:
    return run_optimized_kernel(inputs, weight)
```

**BAD**: Several nested calls wrap across multiple lines, making the order of operations hard to follow.

```python
output = torch.nn.functional.linear(
    inputs,
    torch.mul(
        torch.sub(
            quantized_weight.to(dtype=compute_dtype),
            weight_zero_point.reshape(num_output_features, 1),
        ),
        weight_scale.reshape(num_output_features, 1),
    ),
    bias=bias,
)
```

**GOOD**: Name the intermediate results so each step can be read independently.

```python
converted_weight = quantized_weight.to(dtype=compute_dtype)
centered_weight = torch.sub(converted_weight, weight_zero_point.reshape(num_output_features, 1))
dequantized_weight = torch.mul(centered_weight, weight_scale.reshape(num_output_features, 1))

output = torch.nn.functional.linear(inputs, dequantized_weight, bias=bias)
```

### Use Descriptive Names

Use clear, descriptive variable and function names. Avoid abbreviations that obscure their meaning.

**BAD**: Bare nouns make an action look like a value and a boolean look like an object.

```python
def output_shape(inputs, weight):
    return (*inputs.shape[:-1], weight.shape[0])


hadamard = hadamard_block_size is not None and hadamard_block_size > 1

bias = layer.bias is not None
if bias:
    output = output + layer.bias
```

**GOOD**: Use a verb for the operation and a descriptive boolean name with a prefix such as `is_`, `has_`, `should_`, `can_`, or `use_`.

```python
def get_output_shape(inputs, weight):
    return (*inputs.shape[:-1], weight.shape[0])


use_hadamard = hadamard_block_size is not None and hadamard_block_size > 1

has_bias = layer.bias is not None
if has_bias:
    output = output + layer.bias
```

### Separate Logical Blocks

Short, cohesive blocks do not need extra blank lines. Use blank lines when a longer
sequence moves between distinct stages, rather than after every check or assignment.

**BAD**: A long sequence of validation, tensor preparation, and output computation has no visual separation.

```python
if inputs.ndim < 2:
    raise ValueError("Inputs must have at least two dimensions")
if weight.ndim != 2:
    raise ValueError("Weight must have two dimensions")
if inputs.shape[-1] != weight.shape[-1]:
    raise ValueError("Input and weight dimensions must match")
if inputs.device != weight.device:
    raise ValueError("Inputs and weight must be on the same device")
output_shape = (*inputs.shape[:-1], weight.shape[0])
input_matrix = inputs.reshape(-1, inputs.shape[-1]).to(compute_dtype)
input_matrix = input_matrix.contiguous()
weight_matrix = weight.to(compute_dtype)
transposed_weight = weight_matrix.transpose(-1, -2).contiguous()
output = torch.matmul(input_matrix, transposed_weight)
if bias is not None:
    output = output + bias
output = output.reshape(output_shape)
return output.to(inputs.dtype)
```

**GOOD**: Use a single blank line between stages; keep related operations within each stage together.

```python
if inputs.ndim < 2:
    raise ValueError("Inputs must have at least two dimensions")
if weight.ndim != 2:
    raise ValueError("Weight must have two dimensions")
if inputs.shape[-1] != weight.shape[-1]:
    raise ValueError("Input and weight dimensions must match")
if inputs.device != weight.device:
    raise ValueError("Inputs and weight must be on the same device")

output_shape = (*inputs.shape[:-1], weight.shape[0])
input_matrix = inputs.reshape(-1, inputs.shape[-1]).to(compute_dtype)
input_matrix = input_matrix.contiguous()
weight_matrix = weight.to(compute_dtype)
transposed_weight = weight_matrix.transpose(-1, -2).contiguous()

output = torch.matmul(input_matrix, transposed_weight)
if bias is not None:
    output = output + bias
output = output.reshape(output_shape)
return output.to(inputs.dtype)
```

### Reuse Existing Design

Reuse existing framework structures, abstractions, and concepts whenever possible. Avoid introducing new architectural abstractions, framework mechanisms, or domain concepts unless necessary. If one is required, explain the limitation of the existing design and obtain human approval before implementing it. Ordinary implementation work that follows the existing design, such as adding helper functions or local variables, does not require this additional approval.

## Tests

- When adding or modifying tests, reuse the repository's existing testing frameworks, fixtures, helpers, and patterns whenever possible.
- Prefer extending existing test suites over introducing isolated, narrowly scoped tests that are difficult to maintain. For bug fixes and small features, first look for an existing suite or parametrization to add cases to.
- Tests used only for temporary validation or one-off experiments, such as configuration tuning for a particular case, may be placed under `tests/tmp/<session_name>/`, do not add them to version control.

## Commits

Use Conventional Commits with the following format:

    <type>(<scope>): <description>

Every commit must include a `Signed-off-by` trailer using the committer's name and email. Use `git commit -s` to add it.

When committing code you authored, include a `Co-authored-by` trailer
identifying your tool name and model name. The full Git trailer should have this form:

    Co-authored-by: Agent Name (Model Name) <agent-email>

Use the actual tool and model names, along with an appropriate attribution
email address. Do not invent attribution details.

If your contribution consists only of running tests and creating the commit,
do not add yourself as a co-author.
