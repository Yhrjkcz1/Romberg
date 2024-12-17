import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import re
from scipy.integrate import quad
import sympy as sp
import math

# Trapezoidal integration
def trapezoidal_integration(f, a, b, n=100, singular=None, epsilon=1e-6):
    x = np.linspace(a, b, n + 1)
    if singular is not None:
        for s in singular:
            x = np.where(np.isclose(x, s, atol=epsilon), s + epsilon, x)
    y = f(x)
    return np.trapz(y, x)

# Simpson's integration
def simpson_integration(f, a, b, n=100, singular=None, epsilon=1e-6):
    x = np.linspace(a, b, n + 1)
    if singular is not None:
        for s in singular:
            x = np.where(np.isclose(x, s, atol=epsilon), s + epsilon, x)
    y = f(x)
    return simpson(y, x=x)


def custom_romberg_integration(f, a, b, tol=1e-6, singular=None, epsilon=1e-6):
    max_iters = 20
    romberg_table = []
    errors = []
    n = 1  # Number of intervals (2^0 = 1 interval initially)

    # Modify trapezoidal integration to handle singular
    def adjusted_trapezoidal_integration(f, a, b, n):
        x = np.linspace(a, b, n + 1)
        if singular is not None:
            for s in singular:
                x = np.where(np.isclose(x, s, atol=epsilon), s + epsilon, x)
        y = f(x)
        return np.trapz(y, x)

    first_trap_result = adjusted_trapezoidal_integration(f, a, b, n=n)
    romberg_table.append([first_trap_result])

    for i in range(1, max_iters):
        n *= 2  # Double the number of intervals
        # Use the trapezoidal rule for the current iteration
        new_trap_result = adjusted_trapezoidal_integration(f, a, b, n=n)
        new_row = [new_trap_result]

        # Richardson's extrapolation
        for k in range(1, i + 1):
            new_value = new_row[k - 1] + (new_row[k - 1] - romberg_table[i - 1][k - 1]) / (4 ** k - 1)
            new_row.append(new_value)

        romberg_table.append(new_row)
        current_error = abs(new_row[-1] - romberg_table[i - 1][-1])
        errors.append(current_error)

        # Check for convergence
        if current_error < tol:
            print(f"Converged during main loop at iteration {i}")
            break

    return new_row[-1], len(romberg_table), romberg_table, errors


def parse_function_input():
    """
    Parses user input to define a mathematical function,
    identifies singular points or discontinuities,
    and prepares for singularity handling.

    Returns:
        f: Callable function for the input.
        singular_points: List of detected singular points.
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
            func_str = func_str.replace(")(", ")*(").replace("x(", "x*(")

            # Replace allowed functions with their numpy equivalents
            for func, np_func in allowed_functions.items():
                func_str = func_str.replace(func + "(", np_func + "(")

            print(f"Parsed function string: {func_str}")  # Debugging

            # Define the function
            f = lambda x: eval(func_str)

            # Validate the bounds
            a = get_float_input("Enter the lower bound a:\n>> ")
            b = get_float_input("Enter the upper bound b:\n>> ")
            x = sp.symbols('x')
            singular_points=[]
            # Detect singularities or discontinuities in the interval
            singular_points = detect_singularities_combined(func_str, f, a, b, x, tol=1e-3, num_points=10000)

            if singular_points:
                print(f"Detected singular points or discontinuities within [{a}, {b}]: {singular_points}")

            return f, singular_points, func_str,a,b

        except KeyboardInterrupt:
            print("\nInput was interrupted. Please try again.")
            continue

        except Exception as e:
            print(f"Invalid input. Please try again. Error: {e}\n")
            print("Hint: Ensure the function uses 'x' as the variable and valid math syntax.\n")
            continue

def convert_numpy_to_sympy(expr_str):
    """
    Replaces NumPy functions with SymPy functions for parsing.
    :param expr_str: A string expression containing NumPy functions.
    :return: The modified string with NumPy functions replaced by SymPy functions.
    """
    # Mapping between NumPy functions and SymPy functions
    numpy_to_sympy = {
        'np.log': 'log',
        'np.sqrt': 'sqrt',
        'np.exp': 'exp',
        'np.sin': 'sin',
        'np.cos': 'cos',
        'np.tan': 'tan',
    }
    
    for numpy_func, sympy_func in numpy_to_sympy.items():
        expr_str = re.sub(r'\b' + re.escape(numpy_func) + r'\b', sympy_func, expr_str)
    return expr_str

def detect_singularities_sympy(func_input, x):
    """
    Detects symbolic singularities of a function using SymPy.
    :param func_input: The input symbolic expression (as a string).
    :param x: The symbol representing the variable.
    :return: A list of singularities.
    """
    # Replace NumPy functions with SymPy functions
    func_input = convert_numpy_to_sympy(func_input)

    # Try to parse the input symbolic function
    try:
        func = sp.sympify(func_input)
    except sp.SympifyError:
        raise ValueError(f"Unable to parse the symbolic expression: {func_input}")
    
    singularities = set()

    # Check for cases where the denominator is zero (rational functions)
    if func.is_rational_function(x):
        denominator = sp.denom(func)
        result = sp.solveset(denominator, x, domain=sp.S.Reals)
        singularities.update(_extract_discrete_points(result))
    
    # Check for domain restrictions of logarithmic functions (log(x) domain: x > 0)
    if func.has(sp.log):
        log_args = func.atoms(sp.log)
        for log_expr in log_args:
            arg = log_expr.args[0]
            result = sp.solveset(arg <= 0, x, domain=sp.S.Reals)
            singularities.update(_extract_discrete_points(result))
    
    # Check for domain restrictions of square root functions (sqrt(x) domain: x >= 0)
    if func.has(sp.sqrt):
        sqrt_args = func.atoms(sp.sqrt)
        for sqrt_expr in sqrt_args:
            arg = sqrt_expr.args[0]
            result = sp.solveset(arg < 0, x, domain=sp.S.Reals)
            singularities.update(_extract_discrete_points(result))

    # Check for more complex singularities (e.g., poles, branch points)
    try:
        result = sp.singularities(func, x)
        singularities.update(_extract_discrete_points(result))
    except NotImplementedError:
        pass

    # Only return real singularities
    real_singularities = [s for s in singularities if s.is_real]
    return sorted(real_singularities)

def _extract_discrete_points(result):
    """
    Extracts discrete points from the result of a SymPy set.
    :param result: SymPy result (could be FiniteSet, Interval, etc.)
    :return: A set of discrete points.
    """
    if isinstance(result, sp.FiniteSet):
        return result
    elif isinstance(result, sp.Interval):
        # For intervals, we do not return discrete points
        return set()
    else:
        return set()


def detect_singularities_combined(func_input, f, a, b, x, tol=1e-3, num_points=10000):
    """
    Combines symbolic and numerical methods to detect singularities.
    :param func_input: The symbolic input of the function (as a string).
    :param f: The function object (lambda function).
    :param a: The start point of the interval.
    :param b: The end point of the interval.
    :param x: The symbolic variable.
    :param tol: The tolerance for detecting changes.
    :param num_points: The number of points used for numerical detection.
    :return: A list of singularities.
    """
    # First, use SymPy to detect symbolic singularities
    print("func_input:", func_input)
    print("f:", f)
    sympy_singularities = detect_singularities_sympy(func_input, x)
    
    return sympy_singularities

def get_float_input(prompt):
    """
    Gets a floating-point number from the user, supporting special values like pi, e, inf, etc.
    :param prompt: The prompt message for user input.
    :return: The floating-point number or the numerical representation of a special value.
    """
    special_values = {
        "pi": float("3.1415926535"),  # The floating-point value of π
        "e": float("2.71828182"),    # The floating-point value of Euler's number e
        "inf": float("inf"),          # Positive infinity
        "-inf": float("-inf")         # Negative infinity
    }

    while True:
        user_input = input(prompt).strip().lower()
        # Try to match special values
        if user_input in special_values:
            return special_values[user_input]
        try:
            # Convert to float
            value = float(user_input)
            return value
        except ValueError:
            print(f"Invalid input. Please enter a number or a special value (e.g., {', '.join(special_values.keys())}).")


def visualize_integration(segment_functions, singular,a,b):
    """
    Visualize integration results for segmented functions.

    Args:
        segment_functions: List of callable functions for each segment.
        segments: List of float segment boundaries [a0, a1, ..., an].
    """
    print("singular points:",singular)
    # Initialize results
    true_value = 0
    trap_result = []
    simpson_result = []
    romberg_result = []
    romberg_errors = []
    romberg_table = []
    romberg_error_diff=[]
    romberg_true_errors=[]
    h_values = []  # Collect h values here
    epsilon = 1e-4
    # Loop through each segment to perform calculations
    f = segment_functions
    # Compute true value using quad
    singular_list = [float(x) for x in singular]
    segment_true_value, _ = quad(f, a, b,points=singular_list)
    true_value += segment_true_value

    # Perform integration using different methods
    segment_trap_result = [trapezoidal_integration(f, a, b, n=2 ** j,singular=singular_list, epsilon=epsilon) for j in range(1, 11)]
    segment_simpson_result = [simpson_integration(f, a, b, n=2 ** j,singular=singular_list, epsilon=epsilon) for j in range(1, 11)]
    segment_romberg_result, _, segment_romberg_table_segment, segment_romberg_errors = custom_romberg_integration(
        f, a, b, tol=1e-6,singular=singular_list, epsilon=epsilon
    )

    # Append results for this segment
    trap_result.extend(segment_trap_result)
    simpson_result.extend(segment_simpson_result)
    romberg_result.extend([segment_romberg_result])
    romberg_errors.extend(segment_romberg_errors)

    # Merge the segment's romberg_table into the global romberg_table
    romberg_table.extend(segment_romberg_table_segment)

    # Compute and store h values for this segment
    segment_h_values = [(b - a) / 2 ** j for j in range(1, len(segment_romberg_table_segment) + 1)]
    h_values.extend(segment_h_values)  # Combine all segment h values

    # Calculate true errors
    trap_true_errors = [abs(res - true_value) for res in trap_result]
    simpson_true_errors = [abs(res - true_value) for res in simpson_result]
    # print(romberg_table)
    for i, row in enumerate(romberg_table):

        # 计算 difference（对角线元素差的绝对值）
        if i > 0:
            error = abs(romberg_table[i][i] - romberg_table[i-1][i-1])
            romberg_error_diff.extend([error])

        # 计算 true error（对角线元素与真实值的绝对差）
        if i < len(row):
            true_error = abs(romberg_table[i][i] - true_value)
            romberg_true_errors.extend([true_error])
    # print(romberg_true_errors)
    # print(romberg_error_diff)
    # Calculate the differences in error for each method
    trap_error_diff = [abs(trap_true_errors[i] - trap_true_errors[i - 1]) for i in range(1, len(trap_true_errors))]
    simpson_error_diff = [abs(simpson_true_errors[i] - simpson_true_errors[i - 1]) for i in range(1, len(simpson_true_errors))]

    # Plot the Romberg table with h values
    plot_romberg_table(romberg_table, romberg_errors, h_values)
    
    # Display results
    print("True Value:",true_value)
    for i in range(min(len(romberg_true_errors), len(trap_true_errors), len(simpson_true_errors))):
        print(f"Iteration {i + 1}")
        print(f"  Romberg error: {romberg_true_errors[i]:.6e}")
        print(f"  Trapezoidal error: {trap_true_errors[i]:.6e}")
        print(f"  Simpson error: {simpson_true_errors[i]:.6e}")
        print() 
    # print("Romberg Errors:")
    # for i, error in enumerate(romberg_errors, start=1):
    #     print(f"Iteration {i}: Error = {error:.6e}")

    # Create a figure with subplots for error convergence and other visualizations
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    # Left: Error Convergence (Line Chart)
    ax[0].plot(range(1, len(trap_error_diff) + 1), trap_error_diff, label="Trapezoidal Difference", color='orange',
               linestyle='-.')
    ax[0].plot(range(1, len(simpson_error_diff) + 1), simpson_error_diff, label="Simpson Difference", color='green',
               linestyle='--')
    ax[0].plot(range(1, len(romberg_error_diff) + 1), romberg_error_diff, label="Romberg Difference", color='blue')
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Difference")
    ax[0].set_title("Difference Convergence of each iteration")
    ax[0].legend()

    # Middle: Integral Region Visualization
    x = np.linspace(a, b, 500)
    y = segment_functions(x)
    ax[1].plot(x, y, label=f"f(x)", linestyle='-', alpha=0.7)
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


def plot_romberg_table(romberg_table, romberg_errors, h_values):
    """
    Plot Romberg table with values, errors, and h values, adding iteration labels automatically.
    Prevent overlap by dynamically adjusting figure and cell sizes.
    """
    # Determine the maximum number of columns
    max_cols = max(len(row) for row in romberg_table)

    # Prepare headers for the table
    headers = ["Iteration", "h"] + [f"T_{j}" for j in range(max_cols)] + ["Error"]

    # Prepare table content
    table_data = []
    for i, (h, row, error) in enumerate(zip(h_values, romberg_table, romberg_errors)):
        iteration_label = f"Iteration {i + 1}"
        padded_row = [f"{float(val):.6f}" if val is not None else "N/A" for val in row]
        padded_row += [""] * (max_cols - len(padded_row))
        row_data = [iteration_label, f"{h:.6f}"] + padded_row + [f"{error:.2e}"]
        table_data.append(row_data)

    # Dynamically adjust figure size based on data
    num_rows = len(table_data) + 1  # Add 1 for headers
    num_cols = len(headers)
    fig_width = min(20, num_cols * 1.2)  # Scale width per column
    fig_height = min(10, num_rows * 0.5)  # Scale height per row

    # Plot the table with adjusted size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")

    # Adjust table scaling and font
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Reduce font size if necessary
    table.scale(1.1, 1.1)  # Adjust table scaling to fit better

    # Optional: Adjust column widths
    for i in range(num_cols):
        table.auto_set_column_width([i])

    # Add title
    plt.title("Romberg Integration Table with Iterations and Step Sizes", fontsize=12, pad=10)
    plt.tight_layout(pad=1.5)
    plt.show()

if __name__ == "__main__":
    print("Welcome to the Integration Calculator!")
    print("Supported functions: sin, cos, tan, exp, log, sqrt, pi, e")
    functions, singular, func_str,a,b= parse_function_input()  # Get the function from user
    visualize_integration(functions, singular,a,b)