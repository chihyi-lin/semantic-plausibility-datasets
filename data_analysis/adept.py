import json
import pandas as pd
import matplotlib.pyplot as plt
import os

train_path = "../datasets/adept/train-dev-test-split/train.json"
val_path = "../datasets/adept/train-dev-test-split/val.json"
test_path = "../datasets/adept/train-dev-test-split/test.json"


def combine_data(train_path, val_path, test_path):
    """
    Combining the train/val/test datasets into a single dataset for data analysis.
    """

    with open(train_path, "r") as train_file:
        train_data = json.load(train_file)

    with open(val_path, "r") as val_file:
        val_data = json.load(val_file)

    with open(test_path, "r") as test_file:
        test_data = json.load(test_file)
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    train_df["dataset"] = "train"
    val_df["dataset"] = "val"
    test_df["dataset"] = "test"
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return combined_df


def compute_statistics(df):
    """
    Print and plot statistics of the dataset, including overall, modifiers and Mod-N pairs.
    """
    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)
    
    compute_overall(df, output_folder)
    sorted_modifiers = compute_modifiers(df)
    plot_freq_and_ranking(sorted_modifiers, output_folder)
    compute_mod_n_pairs(df)
    print(f'\nPlease checkout the "plots" folder for label distribution and ranked modifers counts.')

    return None


def compute_overall(df, output_folder):
    """
    Counts total number of instances and label distribution.
    """

    print(f"----- OVERALL -----")
    instance_counts = len(df)
    
    label_mapping = {
    0: "Impossible",
    1: "Less likely",
    2: "Equally likely",
    3: "More likely",
    4: "Necessarily true"
}
    df["label_strings"] = df["label"].map(label_mapping)
    label_counts = df["label_strings"].value_counts()
    label_proportions = label_counts / instance_counts
    label_stat = pd.DataFrame({"Label counts": label_counts,
                                "Label proportions": label_proportions})
    print(f"Instance counts: {instance_counts}")
    print(f"\n{label_stat}")

    # plot_label_distribution
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    bars = ax.bar(label_proportions.index, label_proportions, color="skyblue")

    for bar, proportion in zip(bars, label_proportions):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f"{proportion:.2f}", fontsize=10)
    plt.xlabel("Label")
    plt.ylabel("Proportion")
    plt.title("Label Distribution")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    
    output_path = os.path.join(output_folder, "label_distribution.png")
    plt.savefig(output_path)


def compute_modifiers(df):
    """
    Counting numbers of unique modifiers and the print the top 10 frequent modifiers.
    """
    print(f"\n----- MODIFIERS -----")
    num_unique_modifiers = len(df["modifier"].unique())
    print(f"Counts of unique modifiers: {num_unique_modifiers}")
    unique_modifier_counts = df["modifier"].value_counts()
    sorted_modifiers = unique_modifier_counts.sort_values(ascending=False)
    print(f"\nTop 10 frequent modifiers and their counts: \n{sorted_modifiers[:10]}")
    return sorted_modifiers


def plot_freq_and_ranking(sorted_counts, output_folder):
    """
    Plot frequency and ranking for unique modifiers.
    """
    counts_df = pd.DataFrame({
        "modifier": sorted_counts.index,
        "count": sorted_counts.values
    })
    counts_df["ranking"] = counts_df["count"].rank(ascending=False, method="dense")
    plt.figure()
    plt.scatter(counts_df["ranking"], counts_df["count"])
    plt.xlabel('Ranking (descending order of counts)')
    plt.ylabel('Counts')
    plt.title('Unique Modifiers Counts vs Ranking')
    
    output_path = os.path.join(output_folder, "unique_mod_counts_vs_ranking.png")
    plt.savefig(output_path)


def compute_mod_n_pairs(df):
    print(f"\n----- MODIFIER-NOUN PAIRS -----")
    sorted_pair_counts, top10_pairs = __count_freq_and_top10_pairs(df)
    print(f"\nTop 10 frequent pairs and their counts: \n{top10_pairs}")
    df_label_0 = df[df["label"] == 0]   # label 0 = "Impossible"
    sorted_label_0_pair_counts, label_0_top10_pairs = __count_freq_and_top10_pairs(df_label_0)
    print(f'\nTop 10 frequent pairs and their counts in class "Impossible": \n{label_0_top10_pairs}')
    df_label_4 = df[df["label"] == 4]   # label 4 = "Necessarily true"
    sorted_label_4_pair_counts, label_4_top10_pairs = __count_freq_and_top10_pairs(df_label_4)
    print(f'\nTop 10 frequent pairs and their counts in class "Necessarily true": \n{label_4_top10_pairs}')


def __count_freq_and_top10_pairs(df):
    """
    Count the frequency and determine the top 10 ranking of Modifier-Noun pairs based on their counts.
    """
    pair_counts = df.groupby(["modifier","noun"]).size().reset_index(name="pair_counts")
    sorted_pair_counts = pair_counts.sort_values(by="pair_counts", ascending=False)
    top10_pairs = sorted_pair_counts.head(10)
    return sorted_pair_counts, top10_pairs


combined_df = combine_data(train_path, val_path, test_path)
compute_statistics(combined_df) 
# TODO: main()
# TODO: shellscript()