from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from munkres import Munkres

app = Flask(__name__)

# --- Helper Functions ---

def get_numeric_input_web(value):
    """Validates numeric input from web form."""
    try:
        value = int(value)
        if value <= 0:
            return None, "Please enter a positive integer."
        return value, None
    except ValueError:
        return None, "Invalid input! Please enter a valid integer."

def get_matrix_from_form(form, num_rows, num_cols):
    """Extracts matrix data from the web form."""
    matrix = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            value = form.get(f'cell_{i}_{j}')
            try:
                row.append(int(value))
            except ValueError:
                return None, f"Invalid numeric input at row {i+1}, column {j+1}."
        if len(row) != num_cols:
            return None, f"Incorrect number of values in row {i+1}."
        matrix.append(row)
    return np.array(matrix), None

def get_constraint_matrix_from_form(form, num_rows, num_cols, valid_values, constraint_type):
    """Extracts constraint matrix data from the web form."""
    matrix = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            value = form.get(f'cell_{constraint_type}_{i}_{j}')
            try:
                val = int(value)
                if val not in valid_values:
                    return None, f"Invalid value '{value}' at row {i+1}, column {j+1} in {constraint_type} matrix. Use {valid_values}."
                row.append(val)
            except ValueError:
                return None, f"Invalid numeric input at row {i+1}, column {j+1} in {constraint_type} matrix."
        if len(row) != num_cols:
            return None, f"Incorrect number of values in row {i+1} in {constraint_type} matrix."
        matrix.append(row)
    return np.array(matrix), None

def balance_matrix(cost_matrix, skills, availability, preferences):
    """Balances the matrices for unbalanced worker-job cases."""
    num_workers, num_jobs = cost_matrix.shape

    if num_workers < num_jobs:
        diff = num_jobs - num_workers
        cost_matrix = np.vstack((cost_matrix, np.zeros((diff, num_jobs))))
        skills = np.vstack((skills, np.ones((diff, num_jobs))))
        availability = np.vstack((availability, np.ones((diff, num_jobs))))
        preferences = np.vstack((preferences, np.zeros((diff, num_jobs))))
    elif num_workers > num_jobs:
        diff = num_workers - num_jobs
        cost_matrix = np.hstack((cost_matrix, np.zeros((num_workers, diff))))
        skills = np.hstack((skills, np.ones((num_workers, diff))))
        availability = np.hstack((availability, np.ones((num_workers, diff))))
        preferences = np.hstack((preferences, np.zeros((num_workers, diff))))

    return cost_matrix, skills, availability, preferences

def apply_constraints(cost_matrix, skills, availability, preferences):
    """Applies constraints to the cost matrix."""
    penalty = 999
    dislike_penalty = 10
    preference_bonus = -5

    adjusted_cost = np.copy(cost_matrix)
    adjusted_cost[skills == 0] = penalty
    adjusted_cost[availability == 0] = penalty
    adjusted_cost[preferences == -1] += dislike_penalty
    adjusted_cost[preferences == 1] += preference_bonus

    return adjusted_cost

def solve_assignment(adjusted_cost_matrix):
    """Solves the assignment problem using the Hungarian Algorithm."""
    m = Munkres()
    return m.compute(adjusted_cost_matrix.tolist())

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_workers, err_workers = get_numeric_input_web(request.form['num_workers'])
        num_jobs, err_jobs = get_numeric_input_web(request.form['num_jobs'])
        is_profit = request.form.get('matrix_type') == 'profit'

        if err_workers:
            return render_template('index.html', error_workers=err_workers)
        if err_jobs:
            return render_template('index.html', error_jobs=err_jobs)

        return render_template('input_matrices.html', num_workers=num_workers, num_jobs=num_jobs, is_profit=is_profit, matrix_values={})
    return render_template('index.html')

@app.route('/input', methods=['POST'])
def input_matrices():
    num_workers = int(request.form['num_workers'])
    num_jobs = int(request.form['num_jobs'])
    is_profit = request.form['is_profit'] == 'True'

    profit_or_cost_matrix, err_matrix = get_matrix_from_form(request.form, num_workers, num_jobs)
    if err_matrix:
        return render_template('input_matrices.html', num_workers=num_workers, num_jobs=num_jobs, is_profit=is_profit, error_matrix=err_matrix, matrix_values=request.form)

    skills, err_skills = get_constraint_matrix_from_form(request.form, num_workers, num_jobs, {0, 1}, 'skills')
    if err_skills:
        return render_template('input_matrices.html', num_workers=num_workers, num_jobs=num_jobs, is_profit=is_profit, error_skills=err_skills, matrix_values=request.form)

    availability, err_availability = get_constraint_matrix_from_form(request.form, num_workers, num_jobs, {0, 1}, 'availability')
    if err_availability:
        return render_template('input_matrices.html', num_workers=num_workers, num_jobs=num_jobs, is_profit=is_profit, error_availability=err_availability, matrix_values=request.form)

    preferences, err_preferences = get_constraint_matrix_from_form(request.form, num_workers, num_jobs, {-1, 0, 1}, 'preferences')
    if err_preferences:
        return render_template('input_matrices.html', num_workers=num_workers, num_jobs=num_jobs, is_profit=is_profit, error_preferences=err_preferences, matrix_values=request.form)

    if profit_or_cost_matrix is not None and skills is not None and availability is not None and preferences is not None:
        original_profit_matrix = np.copy(profit_or_cost_matrix) if is_profit else None
        cost_matrix = np.max(profit_or_cost_matrix) - profit_or_cost_matrix if is_profit else profit_or_cost_matrix

        cost_matrix, skills, availability, preferences = balance_matrix(cost_matrix, skills, availability, preferences)
        adjusted_cost_matrix = apply_constraints(cost_matrix, skills, availability, preferences)
        assignments = solve_assignment(adjusted_cost_matrix)

        num_original_workers = int(request.form['num_workers'])
        num_original_jobs = int(request.form['num_jobs'])
        valid_assignments = [(row, col) for row, col in assignments if row < num_original_workers and col < num_original_jobs]

        total_value = 0
        assignment_details = []
        for row, col in valid_assignments:
            if is_profit:
                profit = original_profit_matrix[row][col]
                total_value += profit
                assignment_details.append(f"Worker {row+1} is assigned to Task {chr(65+col)} (Profit: {profit})")
            else:
                cost = profit_or_cost_matrix[row][col]
                total_value += cost
                assignment_details.append(f"Worker {row+1} is assigned to Task {chr(65+col)} (Cost: {cost})")

        matrix_type_str = "Profit" if is_profit else "Cost"
        total_label = f"Total {matrix_type_str} with Constraints"

        return render_template('results.html', assignments=assignment_details, total_label=total_label, total_value=total_value)

    return "Error processing inputs." # Should not reach here if validation is correct

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=10000)