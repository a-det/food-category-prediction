import pandas as pd

INPUT_CSV = "../data/nutrition_with_general_category_top5.csv"
OUTPUT_CSV = "../data/nutrition_with_general_category_top5_RU.csv"


def print_dataset_stats(csv_path):
    print(f"Stats for {csv_path}:")
    df = pd.read_csv(csv_path)

    # Get unique values from the 'general_category' column
    unique_categories = df["General_category"].unique()
    print(unique_categories)

    # Print the count of rows for each unique value in 'General_category'
    category_counts = df["General_category"].value_counts()
    print(category_counts)
    print("-" * 60)


def save_top_k_categories(input_csv, output_csv, k):
    df = pd.read_csv(input_csv)
    top_k = df["General_category"].value_counts().nlargest(k).index
    subset = df[df["General_category"].isin(top_k)]
    subset.to_csv(output_csv, index=False)


def random_undersampling(input_csv, output_csv, random_state=None):
    df = pd.read_csv(input_csv)
    min_count = df["General_category"].value_counts().min()
    undersampled = (
        df.groupby("General_category", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=random_state))
        .reset_index(drop=True)
    )
    undersampled.to_csv(output_csv, index=False)


# Print the situation before processing
print_dataset_stats(INPUT_CSV)


# save_top_k_categories(
#     INPUT_CSV,
#     OUTPUT_CSV,
#     5,
# )

# random_undersampling(
#     INPUT_CSV,
#     OUTPUT_CSV,
# )


# Check results after processing
print_dataset_stats(OUTPUT_CSV)

print_dataset_stats("../data/nutrition_with_general_category.csv")
