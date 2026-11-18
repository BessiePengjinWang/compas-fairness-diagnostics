import matplotlib.pyplot as plt


def plot_gamma_sweep(gamma_sweep, unconstrained_accuracy, unconstrained_p_rule, majority_accuracy, output_path=None):
    figure, accuracy_axis = plt.subplots(figsize=(8, 5), facecolor="white")
    accuracy_axis.set_facecolor("white")
    accuracy_axis.set_xlabel("Gamma (share of accuracy the analyst accepts to lose)")
    accuracy_axis.set_ylabel("Accuracy on test rows", color="tab:blue")
    accuracy_axis.tick_params(axis="y", labelcolor="tab:blue")
    accuracy_line, = accuracy_axis.plot(gamma_sweep["gamma"], gamma_sweep["accuracy"], color="tab:blue", label="Accuracy with fairness threshold")
    unconstrained_accuracy_line = accuracy_axis.axhline(unconstrained_accuracy, color="tab:blue", linestyle=":", label="Accuracy without constraint")
    majority_line = accuracy_axis.axhline(majority_accuracy, color="grey", linestyle="--", label="Always predict the majority class")

    p_rule_axis = accuracy_axis.twinx()
    p_rule_axis.set_ylabel("P-rule (%)", color="tab:red")
    p_rule_axis.tick_params(axis="y", labelcolor="tab:red")
    p_rule_line, = p_rule_axis.plot(gamma_sweep["gamma"], gamma_sweep["p_rule"], color="tab:red", label="P-rule with fairness threshold")
    unconstrained_p_rule_line = p_rule_axis.axhline(unconstrained_p_rule, color="tab:red", linestyle=":", label="P-rule without constraint")
    target_line = p_rule_axis.axhline(80, color="tab:red", linestyle="--", alpha=0.5, label="80% p-rule")

    handles = [accuracy_line, unconstrained_accuracy_line, majority_line, p_rule_line, unconstrained_p_rule_line, target_line]
    accuracy_axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=False)
    accuracy_axis.set_title("Accuracy and p-rule versus gamma (race)")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    return figure
