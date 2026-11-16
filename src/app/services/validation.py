import pandas as pd


def validate_dataset(file_path: str, rules: list) -> dict:
    """Validate a CSV dataset against user-defined rules."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset contains no columns") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Unable to read dataset file") from exc

    results = []

    for rule in rules:
        rule_type = rule.type
        column = rule.column

        if rule_type == "required_column":
            passed = column in df.columns

            results.append({
                "type": rule_type,
                "column": column,
                "passed": passed,
                "message": (
                    f"Required column '{column}' exists."
                    if passed
                    else f"Required column '{column}' is missing."
                ),
            })

        elif column not in df.columns:
            results.append({
                "type": rule_type,
                "column": column,
                "passed": False,
                "message": f"Column '{column}' does not exist.",
            })

        elif rule_type == "numeric_range":
            numeric_values = pd.to_numeric(
                df[column], errors="coerce"
            )
            invalid = numeric_values.isna()

            if rule.minimum is not None:
                invalid |= numeric_values < rule.minimum

            if rule.maximum is not None:
                invalid |= numeric_values > rule.maximum

            invalid_count = int(invalid.sum())
            passed = invalid_count == 0

            results.append({
                "type": rule_type,
                "column": column,
                "passed": passed,
                "invalid_rows": invalid_count,
                "message": (
                    f"All values in '{column}' satisfy the numeric range."
                    if passed
                    else (
                        f"{invalid_count} row(s) in '{column}' "
                        "violate the numeric range or are non-numeric."
                    )
                ),
            })

        elif rule_type == "allowed_values":
            invalid = ~df[column].isin(rule.values)
            invalid_count = int(invalid.sum())
            passed = invalid_count == 0

            results.append({
                "type": rule_type,
                "column": column,
                "passed": passed,
                "invalid_rows": invalid_count,
                "message": (
                    f"All values in '{column}' are allowed."
                    if passed
                    else (
                        f"{invalid_count} row(s) in '{column}' "
                        "contain values outside the allowed list."
                    )
                ),
            })

    passed_count = sum(result["passed"] for result in results)
    total_count = len(results)

    return {
        "total_rules": total_count,
        "passed_rules": passed_count,
        "failed_rules": total_count - passed_count,
        "all_passed": passed_count == total_count,
        "results": results,
    }