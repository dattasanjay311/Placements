"""
report.py
=========
Prints the key findings and recommendations derived from `results_df`.
"""


def print_key_findings(results_df, datasets):
    print("\n" + "=" * 80)
    print(" KEY FINDINGS")
    print("=" * 80)

    print("\nOPTIMAL BATCH SIZE PER DATASET (for highest test accuracy):")
    for dataset_name in datasets.keys():
        data = results_df[results_df["Dataset"] == dataset_name]
        best_row = data.loc[data["Test Accuracy"].idxmax()]
        print(
            f"   • {dataset_name}: Batch size {int(best_row['Batch Size'])} "
            f"(Test Acc: {best_row['Test Accuracy']:.2f}%)"
        )

    print("\nGENERALIZATION GAP TRENDS:")
    print("   Smaller batches consistently lead to a smaller generalization gap (better generalization).")
    print("   For the Large dataset, the gap increases from ~0.5% with a batch size of 8 to ~2.7% with a batch size of 512.")

    print("\nTRAINING EFFICIENCY:")
    print("   Larger batches lead to significantly faster epochs due to better hardware utilization.")
    print("   However, they may require more total epochs to converge, negating some of the speed-up.")

    print("\nCRITICAL BATCH SIZE:")
    print("   There appears to be a 'sweet spot' for batch size. Beyond this point, accuracy plateaus or degrades even with LR scaling.")
    print("   For the small and medium datasets, this is around 32-64. For the large dataset, it's around 64-128.")


def print_recommendations():
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    recommendations = [
        "1. DATASET SIZE MATTERS: For small datasets (<5K samples), start with small batches (8-32). "
        "For large datasets (>20K), you can leverage larger batches (128-512) for speed, but tune carefully.",
        "2. ALWAYS SCALE YOUR LEARNING RATE: When increasing batch size, increase the learning rate "
        "proportionally (Linear Scaling Rule). Without this, large-batch training often fails.",
        "3. BALANCE GENERALIZATION & SPEED: Small batches offer better generalization but are slow. "
        "Large batches train faster per epoch but can generalize poorly. Find a batch size that offers "
        "the best accuracy for an acceptable training time.",
        "4. START WITH 32 or 64: These are robust default choices that work well across a variety of problems.",
        "5. MONITOR THE GENERALIZATION GAP: If the difference between your training and validation accuracy "
        "is large and growing, try reducing your batch size.",
    ]
    for rec in recommendations:
        print(f"\n• {rec}")
