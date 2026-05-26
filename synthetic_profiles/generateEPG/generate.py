import argparse
from tensorflow.keras.models import load_model
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys
import shutil


# Set working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))


def generate_random_latent(batch_size=1, n_lanes=6, length=500):
    return np.random.uniform(low=0, high=1, size=(batch_size, n_lanes, length, 1))



# Generate a batch of clean EPGs
def epg_from_df(df, n_lanes=6, epg_length=5000, scan_min=4000, std_dev=4) -> np.ndarray:
    scan_max = scan_min + epg_length
    color_map = {"blue": 0, "green": 1, "yellow": 2, "red": 3, "purple": 4, "orange": 5}
    epg = np.zeros((1, n_lanes, epg_length, 1))

    # Fill in the EPG with Gaussian peaks based on the CSV data
    for _, row in df.iterrows():
        channel = color_map.get(row["Color"])
        center = int(round(row["Scan"]))
        height = row["Height"]

        start = max(center - 4 * std_dev, scan_min)
        end = min(center + 4 * std_dev, scan_max)

        x = np.arange(start, end, dtype=int)
        gaussian = height * np.exp(-0.5 * ((x - center) / std_dev) ** 2)

        epg[0, channel, (x - scan_min).astype(int), 0] += gaussian
    return epg


# Preprocess EPG: subtract baseline and scale
def preprocess_epg(epg: np.ndarray) -> np.ndarray:
    epg_bs = epg.copy()
    # Turns out baselines are not needed for the current implementation
    for channel in range(epg.shape[1]):
        baseline = mode(epg_bs[0, channel, :, 0], keepdims=False).mode
        epg_bs[0, channel, :, 0] -= baseline
    return epg_bs / 100.0


# Undo preprocessing: add baseline and scale up
def undo_preprocess_epg(epg: np.ndarray) -> np.ndarray:
    epg_restored = epg * 100.0
    return epg_restored


def save_epg_data(epg: np.ndarray, save_path: str):
    # Save the EPG numpy array as a .npy file
    np.save(save_path, epg)




# Main function to generate EPGs from CSV files
# and save them in the specified directory
def main(csv_dir, output_dir, batch_size, epg_length, scan_min, use_generator=True, std_dev=4):
    """
    Generates and saves electropherogram (EPG) data using a pre-trained generator model (optional).
    This function processes all CSV files in the specified `csv_dir`.
    For each file, it:
      - Loads and preprocesses the EPG data.
      - Optionally passes it through the GAN generator model for realism.
      - Saves the generated EPGs as .npy files in `output_dir/epgs`.
    """
    if use_generator:
        generator = load_model('TGAN_model/Profile_generator_GAN_trained_model_random_24-09-2023.h5')
    # Find all relevant CSV files
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    n_files = len(csv_files)
    if n_files == 0:
        print(f"No CSV files found in {csv_dir}")
        return

    # Prepare batch
    dfs = [pd.read_csv(f) for f in csv_files]
    clean_epgs = []
    for df in dfs:
        epg = preprocess_epg(epg_from_df(df, epg_length=epg_length, scan_min=scan_min, std_dev=std_dev))
        clean_epgs.append(epg)
    clean_epgs = np.concatenate(clean_epgs, axis=0)
    latent = generate_random_latent(batch_size=n_files, n_lanes=6, length=500)

    # Output directory for EPGs
    epgs_dir = os.path.join(output_dir, "epgs")
    os.makedirs(epgs_dir, exist_ok=True)

    # Generate EPGs in batches and save as .npy
    for start in range(0, n_files, batch_size):
        end = min(start + batch_size, n_files)
        batch_clean_epgs = clean_epgs[start:end]
        batch_latent = latent[start:end]
        batch_csv_files = csv_files[start:end]
        if use_generator:
            batch_generated_epgs = generator.predict([batch_clean_epgs, batch_latent], batch_size=(end-start))
            # Undo preprocessing after generator
            batch_generated_epgs = np.stack([
                undo_preprocess_epg(epg)
                for epg in batch_generated_epgs
            ])
        else:
            # Undo preprocessing for raw EPGs as well
            batch_generated_epgs = np.stack([
                undo_preprocess_epg(epg)
                for epg in batch_clean_epgs
            ])
        for _, (csv_file, gen_epg) in enumerate(zip(batch_csv_files, batch_generated_epgs)):
            base = os.path.splitext(os.path.basename(csv_file))[0]
            save_path = os.path.join(epgs_dir, base + ".npy")
            save_epg_data(gen_epg, save_path)

    r_output_dir = os.path.join(csv_dir, "..")        
    if r_output_dir is not None:
        # Copy reference_genotypes directory
        src_geno = os.path.join(r_output_dir, "reference_genotypes")
        dst_geno = os.path.join(output_dir, "reference_genotypes")
        if os.path.exists(src_geno):
            if os.path.exists(dst_geno):
                shutil.rmtree(dst_geno)
            shutil.copytree(src_geno, dst_geno)
        # Copy mapping file
        src_map = os.path.join(r_output_dir, "alleles_to_genotypes_mapping.csv")
        dst_map = os.path.join(output_dir, "alleles_to_genotypes_mapping.csv")
        if os.path.exists(src_map):
            shutil.copy2(src_map, dst_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EPGs from CSV files.")
    parser.add_argument("--csv_dir", type=str, default="../generated/generated_alleles/epgs", help="Directory containing input CSV files.")
    parser.add_argument("--output_dir", type=str, default="../generated_epgs/1", help="Directory to save output EPGs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generator model.")
    parser.add_argument("--epg_shape", type=int, nargs=2, metavar=('SCAN_MIN', 'EPG_LENGTH'),
                        # default=[0, 9641],
                        default=[4000, 5000],
                        help="EPG length and scan_min as two integers, e.g. --epg_shape 5000 4000")
    parser.add_argument("--no_generator", action="store_true", help="If set, do not use the GAN generator; output raw EPGs.")
    parser.add_argument("--std_dev", type=float, default=4, help="Standard deviation for Gaussian peaks in EPG generation.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    scan_min, epg_length = args.epg_shape
    main(csv_dir=args.csv_dir, output_dir=args.output_dir, batch_size=args.batch_size, epg_length=epg_length, scan_min=scan_min, use_generator=not args.no_generator, std_dev=args.std_dev)

    # example command:
    # python generate.py --csv_dir='../generated/generated_alleles_20251223_191249/epgs' --output_dir='../generated_epgs/1' --batch_size=64 --epg_shape 4000 5000 --std_dev 4