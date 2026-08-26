# Scale for project Call Me Maybe

## Mandatory Part

### Preliminaries

Check the following requirements:

- Only grade the work that is in the student's or group's Git repository.
- The project must be run using `uv run python -m src`
- All errors should be handled gracefully without crashing
- Verify that the output JSON follows the exact format specified
- Check that constrained decoding is implemented (not just prompting)
- Ensure near-perfect JSON validity (100% parseable output)

### Project Structure and Dependencies

Verify the project setup:

- Run `uv sync` successfully
- Verify that llm_sdk is properly integrated
- Check that all classes use pydantic for validation
- Ensure the program can be run with `uv run python -m src`
- Verify input/ directory structure is correct
- Check that output/ directory is created during execution

### Input File Handling

Test input file processing:

- Verify the program correctly reads `function_calling_tests.json`
- Verify the program correctly reads `functions_definition.json`
- Test with invalid JSON in input files (should handle gracefully)
- Test with missing input files (should provide clear error messages)
- Verify proper error handling without crashes

### Output File Format

Verify output file correctness:

- Check that the output file is created (default: `data/output/function_calling_results.json`, or the path provided with `--output`)
- Verify the file contains 100% valid and retrievable JSON (no syntax or parsing errors)
- Confirm that the JSON strictly follows the expected schema
- Check that each entry includes exactly: `prompt`, `name`, and `parameters` keys
- Ensure all required arguments are present and match the defined schema
- Verify that argument types and allowed values comply with the function specifications (e.g., firmware field restricted to predefined options)
- For parameters whose type is "number", both JSON integers and JSON floating-point values are accepted unless explicitly specified otherwise in the function definition
- Confirm there are no extra keys, text, or prose outside the JSON structure

### Function Calling Accuracy

Evaluate function calling accuracy:

- Test with simple prompts (e.g., "add 2 and 3")
- Verify correct function selection (>90% accuracy expected)
- Check argument extraction accuracy (>90% expected)
- Test with ambiguous prompts
- Verify the system handles edge cases (empty strings, large numbers)

**Does the solution achieve at least 90% accuracy on function selection?**

### LLM SDK Usage

Verify proper LLM SDK usage:

- Check that SDK methods are used according to the SDK implementation
- Ensure no private methods or attributes are accessed
- Confirm the Qwen/Qwen3-0.6B model is used

### Error Handling and Robustness

Test error handling:

- Test with malformed input JSON
- Test with missing function definitions
- Test with prompts that don't match any function
- Verify clear error messages are provided
- Ensure the program never crashes unexpectedly

**Does the program handle all error cases gracefully?**

### Performance and Reliability

Evaluate performance:

- Check that all test prompts are processed in reasonable time (<5 minutes)
- Verify 100% of outputs are valid JSON (parseable)
- Check that the solution achieves >90% accuracy on provided tests
- Verify the system is reliable across multiple runs

**Does the solution meet these performance criteria?**

### Code Quality and Documentation

Review code quality:

- Check that code is well-organized and readable
- Verify proper use of pydantic for validation
- Check that README.md explains the algorithm clearly
- Verify README includes design decisions and challenges
- Check for proper type hints and documentation

**Is the code quality and documentation satisfactory?**
