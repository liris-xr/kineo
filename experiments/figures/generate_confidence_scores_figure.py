import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import os
import argparse

def generate_confidence_scores_figure(output_path):
    input_image = "confidence_score_egohumans.png"
    fp = os.path.join(os.path.dirname(__file__), input_image)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Input image {fp} not found")

    img = mpimg.imread(fp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Confidence (Yellow = High, Dark Purple = Low)', rotation=270, labelpad=25)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="Path to output file")
    args = parser.parse_args()
    output_path = args.output_path
    generate_confidence_scores_figure(output_path)