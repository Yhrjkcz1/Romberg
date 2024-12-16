import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import simpson
import re
from scipy.integrate import quad


# Trapezoidal integration
def trapezoidal_integration(f, a, b, n=100):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return np.trapz(y, x)


# Simpson's integration
def simpson_integration(f, a, b, n=100):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return simpson(y, x=x)

# Modified trapezoidal integration to handle segmented functions
def trapezoidal_integration_segments(segment_functions, segments, n=100):
    total_result = 0
    for i in range(len(segments) - 1):
        f_segment = segment_functions[i]
        a, b = segments[i], segments[i + 1]
        total_result += trapezoidal_integration(f_segment, a, b, n)
    return total_result

# Modified Simpson's integration to handle segmented functions
def simpson_integration_segments(segment_functions, segments, n=100):
    total_result = 0
    for i in range(len(segments) - 1):
        f_segment = segment_functions[i]
        a, b = segments[i], segments[i + 1]
        total_result += simpson_integration(f_segment, a, b, n)
    return total_result
def custom_romberg_integration(f, a, b, tol=1e-6):
    """
    Perform Romberg integration with early stopping during both the main iteration
    and the Richardson extrapolation phase.
    """
    max_iters = 20
    romberg_table = []
    errors = []
    h = b - a
    romberg_table.append([0.5 * h * (f(a) + f(b))])  # First trapezoidal rule

    for i in range(1, max_iters):
        h /= 2
        # Trapezoidal rule refinement
        sum_trapezoids = sum(f(a + k * h) for k in range(1, 2 ** i, 2))
        new_row = [0.5 * romberg_table[i - 1][0] + h * sum_trapezoids]

        # Richardson's extrapolation with early stopping
        for k in range(1, i + 1):
            new_value = new_row[k - 1] + (new_row[k - 1] - romberg_table[i - 1][k - 1]) / (4 ** k - 1)
            new_row.append(new_value)

            # # Check for convergence in Richardson extrapolation
            # if abs(new_row[-1] - new_row[-2]) < tol:
            #     print(f"Converged during Richardson extrapolation at iteration {i}, level {k}")
            #     break

        romberg_table.append(new_row)
        current_error = abs(new_row[-1] - romberg_table[i - 1][-1])
        errors.append(current_error)

        # Check for convergence in main loop
        if current_error < tol:
            print(f"Converged during main loop at iteration {i}")
            break

    return new_row[-1], len(romberg_table), romberg_table, errors
def custom_romberg_integration_segments(segment_functions, segments, tol=1e-6):
    """
    Perform Romberg integration for segmented functions with early stopping.
    Each segment is integrated separately, and the results are combined.

    Args:
        segment_functions: List of callable functions for each segment.
        segments: List of float segment boundaries [a0, a1, ..., an].
        tol: Tolerance for convergence.

    Returns:
        total_result: Combined integral result for all segments.
        total_iters: Total number of iterations across all segments.
        all_romberg_tables: List of Romberg tables for each segment.
        all_errors: Combined error list across all segments.
    """
    total_result = 0
    total_iters = 0
    all_romberg_tables = []
    all_errors = []

    # Loop through each segment
    for i in range(len(segments) - 1):
        f = segment_functions[i]
        a, b = segments[i], segments[i + 1]

        max_iters = 20
        romberg_table = []
        errors = []
        h = b - a
        romberg_table.append([0.5 * h * (f(a) + f(b))])  # First trapezoidal rule

        for j in range(1, max_iters):
            h /= 2
            # Trapezoidal rule refinement
            sum_trapezoids = sum(f(a + k * h) for k in range(1, 2 ** j, 2))
            new_row = [0.5 * romberg_table[j - 1][0] + h * sum_trapezoids]

            # Richardson's extrapolation with early stopping
            for k in range(1, j + 1):
                new_value = new_row[k - 1] + (new_row[k - 1] - romberg_table[j - 1][k - 1]) / (4 ** k - 1)
                new_row.append(new_value)

                # Check for convergence in Richardson extrapolation
                if abs(new_row[-1] - new_row[-2]) < tol:
                    print(f"Segment {i + 1}: Converged during Richardson extrapolation at iteration {j}, level {k}")
                    break

            romberg_table.append(new_row)
            current_error = abs(new_row[-1] - romberg_table[j - 1][-1])
            errors.append(current_error)

            # Check for convergence in main loop
            if current_error < tol:
                print(f"Segment {i + 1}: Converged during main loop at iteration {j}")
                break

        # Aggregate results
        total_result += new_row[-1]
        total_iters += len(romberg_table)
        all_romberg_tables.append(romberg_table)
        all_errors.extend(errors)

    return total_result, total_iters, all_romberg_tables, all_errors


def parse_function_input():
    """
    Parses user input to define a mathematical function,
    identifies singular points or discontinuities,
    and prepares segmented functions and intervals if necessary.

    Returns:
        segment_functions: List of callable functions for each segment.
        segments: List of interval boundaries defining segments.
        func_str: String representation of the original input function.
    """
    allowed_functions = {
        "sin": "np.sin",
        "cos": "np.cos",
        "tan": "np.tan",
        "exp": "np.exp",
        "log": "np.log",
        "sqrt": "np.sqrt",
        "pi": "np.pi",
        "e": "np.e"
    }

    while True:
        try:
            # Prompt the user for the function
            func_str = input("Enter a function f(x) (e.g., 'sin(x)', 'x**2 + exp(x)', 'sqrt(x) + log(x)'):\n>> ")

            # Ensure multiplication is explicit between constants and variables
            func_str = func_str.replace(")(", ")*(")
            func_str = func_str.replace("x(", "x*(")

            # Replace allowed functions with their numpy equivalents
            for func, np_func in allowed_functions.items():
                func_str = func_str.replace(func + "(", np_func + "(")

            print(f"Parsed function string: {func_str}")  # Debugging

            # Define the function
            f = lambda x: eval(func_str)

            # Validate the bounds
            a = get_float_input("Enter the lower bound a:\n>> ")
            b = get_float_input("Enter the upper bound b:\n>> ")

            # Detect singularities or discontinuities in the interval
            singular_points = []
            test_points = np.linspace(a, b, 1000)  # Fine sampling of the interval
            for x in test_points:
                try:
                    _ = f(x)  # Test function evaluation
                except:
                    singular_points.append(x)

            # Remove duplicate singularities and sort them
            singular_points = sorted(set(singular_points))

            if singular_points:
                print(f"Detected singular points or discontinuities within [{a}, {b}]: {singular_points}")

                # Create segments
                segments = [a] + singular_points + [b]
                print(f"The integral will be divided into {len(segments) - 1} segments: {segments}")

                # Define segment-specific functions
                segment_functions = []
                for i in range(len(segments) - 1):
                    x_start, x_end = segments[i], segments[i + 1]

                    # Create a piecewise function valid in each segment
                    segment_functions.append(
                        lambda x, f=f, x_start=x_start, x_end=x_end: f(x) if x_start <= x <= x_end else 0
                    )

                return segment_functions, segments, func_str

            else:
                # No singularities: treat as a single segment
                return [f], [a, b], func_str

        except KeyboardInterrupt:
            print("\nInput was interrupted. Please try again.")
            continue

        except Exception as e:
            print(f"Invalid input. Please try again. Error: {e}\n")
            print("Hint: Ensure the function uses 'x' as the variable and valid math syntax.\n")
            continue


def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")


def visualize_integration(segment_functions, segments):
    """
    Visualize integration results for segmented functions.

    Args:
        segment_functions: List of callable functions for each segment.
        segments: List of float segment boundaries [a0, a1, ..., an].
    """
    # Initialize results
    true_value = 0
    trap_result = []
    simpson_result = []
    romberg_result = 0
    romberg_errors = []
    romberg_table = []

    # Loop through each segment to perform calculations
    for i in range(len(segments) - 1):
        f = segment_functions[i]
        a, b = segments[i], segments[i + 1]

        # Compute true value using quad
        segment_true_value, _ = quad(f, a, b)
        true_value += segment_true_value

        # Perform integration using different methods
        segment_trap_result = [trapezoidal_integration(f, a, b, n=2 ** j) for j in range(1, 11)]
        segment_simpson_result = [simpson_integration(f, a, b, n=2 ** j) for j in range(1, 11)]
        segment_romberg_result, _, segment_romberg_table_segment, segment_romberg_errors = custom_romberg_integration(
            f, a, b, tol=1e-6
        )

        # Append results for this segment
        trap_result.extend(segment_trap_result)
        simpson_result.extend(segment_simpson_result)
        romberg_result += segment_romberg_result
        romberg_errors.extend(segment_romberg_errors)

        # Merge the segment's romberg_table into the global romberg_table
        romberg_table.extend(segment_romberg_table_segment)

    # Calculate true errors
    trap_true_errors = [abs(res - true_value) for res in trap_result]
    simpson_true_errors = [abs(res - true_value) for res in simpson_result]

    # Fix: Correctly calculate Romberg errors based on the last column of each table row
    romberg_true_errors = []
    for table in romberg_table:
        if isinstance(table, list):  # Ensure that each table is a list
            romberg_true_errors.append(abs(table[-1] - true_value))  # Last column of each row in the table

    plot_romberg_table(romberg_table, romberg_errors)

    # Display results
    print(f"True Value: {true_value:.6f}")
    print(f"Romberg Result: {romberg_result:.6f}")
    print("Romberg Errors:")
    for i, error in enumerate(romberg_errors, start=1):
        print(f"Iteration {i}: Error = {error:.6e}")

    # Create a figure with subplots for error convergence and other visualizations
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    # Left: Error Convergence (Line Chart)
    ax[0].plot(range(1, len(trap_true_errors) + 1), trap_true_errors, label="Trapezoidal True Error", color='orange',
               linestyle='-.')
    ax[0].plot(range(1, len(simpson_true_errors) + 1), simpson_true_errors, label="Simpson True Error", color='green',
               linestyle='--')
    ax[0].plot(range(1, len(romberg_true_errors) + 1), romberg_true_errors, label="Romberg True Error", color='blue')
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Error")
    ax[0].set_title("Error Convergence Relative to True Value")
    ax[0].legend()

    # Middle: Integral Region Visualization
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        x = np.linspace(a, b, 500)
        y = segment_functions[i](x)
        ax[1].plot(x, y, label=f"f(x) Segment {i + 1}", linestyle='-', alpha=0.7)
        ax[1].fill_between(x, y, where=[(a <= xi <= b) for xi in x], alpha=0.3)

    ax[1].set_xlabel("x")
    ax[1].set_ylabel("f(x)")
    ax[1].set_title("Integration Area")
    ax[1].legend()

    # Right: Error convergence per method
    ax[2].plot(range(1, len(trap_true_errors) + 1), trap_true_errors, label="Trapezoidal True Error", color='orange',
               linestyle='-.')
    ax[2].plot(range(1, len(simpson_true_errors) + 1), simpson_true_errors, label="Simpson True Error", color='green',
               linestyle='--')
    ax[2].plot(range(1, len(romberg_true_errors) + 1), romberg_true_errors, label="Romberg True Error", color='blue')
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Error")
    ax[2].set_title("Error Relative to True Value")
    ax[2].legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_romberg_table(romberg_table, romberg_errors):
    """
    Plot Romberg table as a simple table with values and errors.

    Args:
        romberg_table: List of lists, where each inner list is a row of the Romberg table.
        romberg_errors: List of errors corresponding to each row in the Romberg table.
    """
    # Determine the maximum number of columns
    max_cols = max(len(row) for row in romberg_table)

    # Prepare headers for the table
    headers = ["Iteration"] + [f"T_{j}" for j in range(max_cols)] + ["Error"]

    # Prepare table content
    table_data = []
    for i, (row, error) in enumerate(zip(romberg_table, romberg_errors), start=1):
        # Format each row while handling incompatible elements
        padded_row = []
        for val in row:
            try:
                # Attempt to format as float
                padded_row.append(f"{float(val):.6f}")
            except (ValueError, TypeError):
                # If formatting fails, add placeholder
                padded_row.append("N/A")

        # Pad the row to match the maximum number of columns
        padded_row += [""] * (max_cols - len(padded_row))

        # Add iteration label and error column
        row_data = [f"Iter {i}"] + padded_row + [f"{error:.2e}"]
        table_data.append(row_data)

    # Plot the table
    fig, ax = plt.subplots(figsize=(12, len(romberg_table) + 2))  # Adjust height based on number of rows
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")

    # Adjust table font size and layout
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))

    # Add title
    plt.title("Romberg Integration Table", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()



def validate_input(f, func_str):

    def extract_conditions(func_str):
        conditions = []

        # Match patterns for log(x), log(sin(x)), sqrt(x), and division
        log_matches = re.finditer(r"log\(([^()]+|[^()]*\([^()]*\)[^()]*)\)", func_str)
        sqrt_matches = re.finditer(r"sqrt\(([^()]+|[^()]*\([^()]*\)[^()]*)\)", func_str)
        division_matches = re.finditer(r"/\(([^()]+|[^()]*\([^()]*\)[^()]*)\)", func_str)

        for match in log_matches:
            arg = match.group(1).strip()
            conditions.append(f"({arg}) > 0")  # log requires arg > 0

        for match in sqrt_matches:
            arg = match.group(1).strip()
            conditions.append(f"({arg}) >= 0")  # sqrt requires arg >= 0

        for match in division_matches:
            arg = match.group(1).strip()
            conditions.append(f"({arg}) != 0")  # Denominator cannot be zero

        return conditions

    # Extract domain conditions
    domain_conditions = extract_conditions(func_str)
    if domain_conditions:
        print(f"Detected domain constraints: {', '.join(domain_conditions)}")

    while True:
        try:
            # Get integration limits from the user
            a = float(input("Enter the lower limit of integration (a): "))
            b = float(input("Enter the upper limit of integration (b): "))

            # Check all domain constraints for both a and b
            invalid = False
            for condition in domain_conditions:
                # Replace 'x' in the condition with the current limit
                condition_a = condition.replace('x', str(a))
                condition_b = condition.replace('x', str(b))

                # Evaluate the condition dynamically
                if not eval(condition_a) or not eval(condition_b):
                    print(f"Invalid input: {condition} not satisfied for a={a} or b={b}. Please try again.")
                    invalid = True
                    break

            if not invalid:
                return a, b  # If all checks pass, return the valid limits

        except ValueError:
            print("Invalid input. Please enter numeric values for a and b.\n")
        except Exception as e:
            print(f"Unexpected error: {e}. Please check your input and try again.\n")

if __name__ == "__main__":
    print("Welcome to the Integration Calculator!")
    print("Supported functions: sin, cos, tan, exp, log, sqrt, pi, e")
    segment_functions, segments, func_str= parse_function_input()  # Get the function from user
    visualize_integration(segment_functions, segments)

def simpson_romberg_integration(f, a, b, tol=1e-6, max_iter=20):
    """
    Perform Romberg integration using Simpson's rule as the base method.
    """
    R = np.zeros((max_iter, max_iter))

    # Compute R[0, 0] using Simpson's rule with 2 intervals (n=2)
    R[0, 0] = simpson_integration(f, a, b, n=2)
    errors = []

    for i in range(1, max_iter):
        # Double the number of intervals for Simpson's rule
        n_intervals = 2 ** (i + 1)
        R[i, 0] = simpson_integration(f, a, b, n=n_intervals)

        # Perform Richardson extrapolation
        for j in range(1, i + 1):
            R[i, j] = (4 ** j * R[i, j - 1] - R[i - 1, j - 1]) / (4 ** j - 1)

        # Compute error
        current_error = abs(R[i, i] - R[i - 1, i - 1])
        errors.append(current_error)

        # Check for convergence
        if current_error < tol:
            return R[i, i], i + 1, R[:i + 1, :i + 1], errors

    # Return the final result if maximum iterations are reached
    return R[max_iter - 1, max_iter - 1], max_iter, R, errors

