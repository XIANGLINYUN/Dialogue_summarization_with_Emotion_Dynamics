"""
Runner for the IEMOCAP pipeline.
Executes preprocessing → segmentation → evaluation in sequence.
"""

from . import preprocess, segmentation, evaluation


def main():
    # print("=== Step 1: Preprocessing ===")
    # preprocess.main()

    print("\n=== Step 2: Segmentation ===")
    segmentation.main()

    # print("\n=== Step 3: Evaluation ===")
    # evaluation.main()

    print("\nSegementation pipeline completed successfully ✅")


if __name__ == "__main__":
    main()
