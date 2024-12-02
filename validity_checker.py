import sympy
from sympy.logic.boolalg import Or, And, Not, Implies
from sympy.abc import S, G, C
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Define premise functions dynamically
def premise1(S, G, C):
    return Or(S, G, C)

def premise2(S, G, C):
    return Implies(G, C)

def premise3(S, G, C):
    return Not(S)

def premise4(S, G, C):
    return And(S, Not(G))

# Conclusion function
def conclusion(S, G, C):
    return C

# Checking for tautology
def is_tautology(expr, truth_combinations):
    return all(expr.subs({S: val[0], G: val[1], C: val[2]}) for val in truth_combinations)

# Checking for contradiction
def is_contradiction(expr, truth_combinations):
    return all(not expr.subs({S: val[0], G: val[1], C: val[2]}) for val in truth_combinations)

# Plotting truth table
def plot_truth_table(truth_combinations, premise_values, conclusion_values):
    num_combinations = len(truth_combinations)  # Total truth combinations
    num_premises = len(premise_values[0])       # Number of premises

    plt.figure(figsize=(12, 6))

    # Premises heatmap
    plt.subplot(1, 2, 1)
    premises_grid = np.array(premise_values).T  # Transpose for proper dimensions
    plt.imshow(premises_grid, cmap='RdYlGn', interpolation='nearest', aspect='auto')
    plt.colorbar(label="Truth Value")
    plt.xticks(range(num_combinations), [f"T{i+1}" for i in range(num_combinations)], rotation=45)
    plt.yticks(range(num_premises), [f"Premise {i+1}" for i in range(num_premises)])
    plt.title("Premises Truth Table")
    # Annotate cells with truth values
    for i in range(num_premises):
        for j in range(num_combinations):
            plt.text(j, i, str(premises_grid[i, j]), ha='center', va='center', color="black")

    # Conclusion heatmap
    plt.subplot(1, 2, 2)
    conclusion_grid = np.array(conclusion_values).reshape(1, -1)
    plt.imshow(conclusion_grid, cmap='RdYlGn', interpolation='nearest', aspect='auto')
    plt.colorbar(label="Truth Value")
    plt.xticks(range(num_combinations), [f"T{i+1}" for i in range(num_combinations)], rotation=45)
    plt.yticks([0], ["Conclusion"])
    plt.title("Conclusion Truth Table")
    # Annotate cells with truth values
    for j in range(num_combinations):
        plt.text(j, 0, str(conclusion_grid[0, j]), ha='center', va='center', color="black")

    plt.tight_layout()
    plt.show()



def validity_checker(premises, conclusion_func):
    truth_combinations = list(product([True, False], repeat=3))  # Truth table combinations for 3 variables (S, G, C)
    validity = True
    premise_values = []
    conclusion_values = []

    # Print header for truth table
    print("S\tG\tC\t" + "\t".join(f"P{i+1}" for i in range(len(premises))) + "\tConclusion")

    for combo in truth_combinations:
        # Map truth values to symbolic variables
        truth_dict = {S: combo[0], G: combo[1], C: combo[2]}

        # Evaluate premises symbolically, then substitute truth values
        evaluated_premises = [premise(S, G, C).subs(truth_dict) for premise in premises]
        evaluated_conclusion = conclusion_func(S, G, C).subs(truth_dict)

        # Convert evaluated results to booleans for analysis and plotting
        premise_bools = [bool(p) for p in evaluated_premises]
        conclusion_bool = bool(evaluated_conclusion)

        # Append to results for plotting
        premise_values.append([int(b) for b in premise_bools])
        conclusion_values.append(int(conclusion_bool))

        # Print truth table row
        print("\t".join(map(str, combo)) + "\t" +
              "\t".join(map(str, premise_bools)) + "\t" + str(conclusion_bool))

        # Check validity: If all premises are true and conclusion is false, the argument is invalid
        if all(premise_bools) and not conclusion_bool:
            validity = False

    # Output validity result
    if validity:
        print("The argument is valid.")
    else:
        print("The argument is invalid.")

    # Plot truth table
    plot_truth_table(truth_combinations, premise_values, conclusion_values)

# Main function
def main():
    premises = [premise1, premise2, premise3, premise4]
    validity_checker(premises, conclusion)

if __name__ == "__main__":
    main()
