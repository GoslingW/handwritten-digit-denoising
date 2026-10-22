import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')  # Must be set before importing matplotlib.pyplot
import matplotlib.pyplot as plt
from improved_dataset import DenoisingDataset, get_transform
from improved_model import ImprovedUNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import seaborn as sns
import os
import json

# Set style
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class ModelAnalyzer:
    def __init__(self, model_path, noise_dir, origin_dir, output_dir='analysis_results',
                 loss_history_path='loss_history.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.output_dir = output_dir
        self.loss_history_path = loss_history_path
        os.makedirs(output_dir, exist_ok=True)

        # Load model
        self.model = ImprovedUNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Prepare dataset
        transform = get_transform()
        full_dataset = DenoisingDataset(
            noise_dir=noise_dir,
            origin_dir=origin_dir,
            transform=transform,
            augment=False
        )

        # Split validation set (consistent with training)
        val_size = 500
        train_size = len(full_dataset) - val_size
        _, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        # Load loss history
        self.loss_history = self.load_loss_history()

    def load_loss_history(self):
        """Load training loss history"""
        if os.path.exists(self.loss_history_path):
            try:
                with open(self.loss_history_path, 'r') as f:
                    loss_history = json.load(f)
                print(f'✓ Loaded loss history: {self.loss_history_path}')
                print(f'  Training epochs: {len(loss_history.get("train", []))}')
                return loss_history
            except Exception as e:
                print(f'✗ Failed to load loss history: {e}')
                return None
        else:
            print(f'⚠️  Loss history file not found: {self.loss_history_path}')
            return None

    def plot_0_training_curve(self):
        """Figure 0: Training curve (Train Loss vs Val Loss)"""
        if self.loss_history is None:
            print('⚠️  Skipping training curve plot (no loss history data)')
            return

        train_losses = self.loss_history.get('train', [])
        val_losses = self.loss_history.get('val', [])

        if not train_losses or not val_losses:
            print('⚠️  Skipping training curve plot (empty loss history data)')
            return

        epochs = range(1, len(train_losses) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left plot: Complete training curve
        ax1 = axes[0]
        ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, color='#2E86AB')
        ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2, color='#A23B72')

        # Mark best validation point
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax1.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5,
                    label=f'Best (Epoch {best_epoch})')
        ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.3)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Curve (Complete)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Right plot: First 20 epochs detail
        ax2 = axes[1]
        max_epoch_detail = min(20, len(train_losses))
        epochs_detail = range(1, max_epoch_detail + 1)
        ax2.plot(epochs_detail, train_losses[:max_epoch_detail],
                 label='Train Loss', linewidth=2, marker='o', markersize=4, color='#2E86AB')
        ax2.plot(epochs_detail, val_losses[:max_epoch_detail],
                 label='Val Loss', linewidth=2, marker='s', markersize=4, color='#A23B72')

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title(f'Training Curve (First {max_epoch_detail} Epochs)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '00_training_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

        # Print training statistics
        print(f'\nTraining Curve Statistics:')
        print(f'  Total epochs: {len(train_losses)}')
        print(f'  Best validation epoch: {best_epoch}')
        print(f'  Best validation loss: {best_val_loss:.4f}')
        print(f'  Final training loss: {train_losses[-1]:.4f}')
        print(f'  Final validation loss: {val_losses[-1]:.4f}')
        print(f'  Training improvement: {train_losses[0]:.4f} → {train_losses[-1]:.4f} '
              f'({(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%)')
        print(f'  Validation improvement: {val_losses[0]:.4f} → {best_val_loss:.4f} '
              f'({(val_losses[0] - best_val_loss) / val_losses[0] * 100:.1f}%)\n')

    def collect_metrics(self):
        """Collect all metrics"""
        print('Collecting performance metrics...')

        psnr_list = []
        ssim_list = []
        samples = []

        with torch.no_grad():
            for i, (noise_img, clean_img) in enumerate(self.val_loader):
                noise_img = noise_img.to(self.device)
                clean_img = clean_img.to(self.device)

                denoised = self.model(noise_img)

                # Convert to numpy
                noise_np = noise_img.squeeze().cpu().numpy()
                denoised_np = denoised.squeeze().cpu().numpy()
                clean_np = clean_img.squeeze().cpu().numpy()

                # Calculate metrics
                psnr_val = psnr(clean_np, denoised_np, data_range=1.0)
                ssim_val = ssim(clean_np, denoised_np, data_range=1.0)

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                # Save samples
                samples.append({
                    'noise': noise_np,
                    'denoised': denoised_np,
                    'origin': clean_np,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                })

                if (i + 1) % 200 == 0:
                    print(f'  Progress: {i + 1}/{len(self.val_loader)}')

        print(f'✓ Collection complete: {len(samples)} samples\n')
        return psnr_list, ssim_list, samples

    def plot_1_metrics_distribution(self, psnr_list, ssim_list):
        """Figure 1: PSNR and SSIM distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PSNR
        ax1 = axes[0]
        mean_psnr = np.mean(psnr_list)
        ax1.hist(psnr_list, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(mean_psnr, color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('PSNR (dB)', fontsize=12)
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.set_title(f'PSNR Distribution\nMean: {mean_psnr:.2f} dB', fontsize=14)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # SSIM
        ax2 = axes[1]
        mean_ssim = np.mean(ssim_list)
        ax2.hist(ssim_list, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(mean_ssim, color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.set_xlabel('SSIM', fontsize=12)
        ax2.set_ylabel('Number of Images', fontsize=12)
        ax2.set_title(f'SSIM Distribution\nMean: {mean_ssim:.4f}', fontsize=14)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '01_metrics_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

    def plot_2_sample_comparison(self, samples, psnr_list):
        """Figure 2: Sample comparison (best/medium/worst)"""
        sorted_indices = np.argsort(psnr_list)

        # Select representative samples
        indices = [
            sorted_indices[-1],  # Best
            sorted_indices[len(sorted_indices) // 2],  # Medium
            sorted_indices[0],  # Worst
        ]

        selected = [samples[idx] for idx in indices]

        fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))
        if len(selected) == 1:
            axes = axes.reshape(1, -1)

        for i, sample in enumerate(selected):
            # Input
            axes[i, 0].imshow(sample['noise'], cmap='gray')
            axes[i, 0].set_title('Input (Noisy)', fontsize=10)
            axes[i, 0].axis('off')

            # Output
            axes[i, 1].imshow(sample['denoised'], cmap='gray')
            axes[i, 1].set_title(f'Model Output (Denoised)\nPSNR: {sample["psnr"]:.2f}\nSSIM: {sample["ssim"]:.3f}', fontsize=10)
            axes[i, 1].axis('off')

            # Ground Truth
            axes[i, 2].imshow(sample['origin'], cmap='gray')
            axes[i, 2].set_title('Ground Truth (Original)', fontsize=10)
            axes[i, 2].axis('off')

            # Residual
            residual = np.abs(sample['denoised'] - sample['origin'])
            im = axes[i, 3].imshow(residual, cmap='hot', vmin=0, vmax=0.5)
            axes[i, 3].set_title('Residual (Difference)', fontsize=10)
            axes[i, 3].axis('off')

        cbar = plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Absolute Error', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '02_sample_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

    def plot_3_correlation(self, psnr_list, ssim_list):
        """Figure 3: PSNR and SSIM correlation"""
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(psnr_list, ssim_list, alpha=0.5, s=20, c=psnr_list, cmap='viridis')

        # Fitting line
        z = np.polyfit(psnr_list, ssim_list, 1)
        p = np.poly1d(z)
        ax.plot(psnr_list, p(psnr_list), "r--", alpha=0.8, linewidth=2, label=f'Fit line: y={z[0]:.4f}x+{z[1]:.4f}')

        ax.set_xlabel('PSNR (dB)', fontsize=12)
        ax.set_ylabel('SSIM', fontsize=12)
        ax.set_title('Correlation between PSNR and SSIM', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='PSNR (dB)')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '03_psnr_ssim_correlation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

    def plot_4_cumulative_distribution(self, psnr_list, ssim_list):
        """Figure 4: Cumulative distribution function"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PSNR cumulative distribution
        ax1 = axes[0]
        sorted_psnr = np.sort(psnr_list)
        cumulative = np.arange(1, len(sorted_psnr) + 1) / len(sorted_psnr)
        ax1.plot(sorted_psnr, cumulative, linewidth=2)
        ax1.set_xlabel('PSNR (dB)', fontsize=12)
        ax1.set_ylabel('Cumulative Probability', fontsize=12)
        ax1.set_title('PSNR Cumulative Distribution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Median')
        ax1.legend()

        # SSIM cumulative distribution
        ax2 = axes[1]
        sorted_ssim = np.sort(ssim_list)
        cumulative = np.arange(1, len(sorted_ssim) + 1) / len(sorted_ssim)
        ax2.plot(sorted_ssim, cumulative, linewidth=2, color='green')
        ax2.set_xlabel('SSIM', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('SSIM Cumulative Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Median')
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '04_cumulative_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

    def plot_5_error_distribution(self, samples):
        """Figure 5: Error distribution analysis"""
        all_errors = []
        for sample in samples:
            error = np.abs(sample['denoised'] - sample['origin'])
            all_errors.extend(error.flatten())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error histogram
        ax1 = axes[0]
        ax1.hist(all_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Absolute Error', fontsize=12)
        ax1.set_ylabel('Number of Pixels', fontsize=12)
        ax1.set_title(f'Pixel-level Error Distribution\nMean Error: {np.mean(all_errors):.4f}', fontsize=14)
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3)

        # Box plot
        ax2 = axes[1]
        error_per_image = [np.mean(np.abs(s['denoised'] - s['origin'])) for s in samples]
        ax2.boxplot(error_per_image, vert=True)
        ax2.set_ylabel('Mean Absolute Error', fontsize=12)
        ax2.set_title('Mean Error Distribution per Image', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '05_error_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

    def plot_6_heatmap(self, samples):
        """Figure 6: Average error heatmap"""
        # Calculate average error map
        H, W = samples[0]['origin'].shape
        avg_error = np.zeros((H, W))

        for sample in samples:
            error = np.abs(sample['denoised'] - sample['origin'])
            avg_error += error

        avg_error /= len(samples)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(avg_error, cmap='hot', interpolation='nearest')
        ax.set_title('Average Error Heatmap\n(Darker color indicates higher error in that region)', fontsize=14)
        ax.axis('off')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mean Absolute Error', fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '06_error_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

    def plot_7_percentile_analysis(self, psnr_list, ssim_list):
        """Figure 7: Percentile analysis"""
        percentiles = [10, 25, 50, 75, 90, 95, 99]

        psnr_percentiles = [np.percentile(psnr_list, p) for p in percentiles]
        ssim_percentiles = [np.percentile(ssim_list, p) for p in percentiles]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PSNR percentiles
        ax1 = axes[0]
        ax1.plot(percentiles, psnr_percentiles, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Percentile', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('PSNR Percentile Analysis', fontsize=14)
        ax1.grid(True, alpha=0.3)
        for i, (p, v) in enumerate(zip(percentiles, psnr_percentiles)):
            ax1.text(p, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

        # SSIM percentiles
        ax2 = axes[1]
        ax2.plot(percentiles, ssim_percentiles, marker='s', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Percentile', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title('SSIM Percentile Analysis', fontsize=14)
        ax2.grid(True, alpha=0.3)
        for i, (p, v) in enumerate(zip(percentiles, ssim_percentiles)):
            ax2.text(p, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, '07_percentile_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved: {save_path}')
        plt.close()

    def print_statistics(self, psnr_list, ssim_list):
        """Print statistics information"""
        print('\n' + '=' * 70)
        print('Detailed Statistics')
        print('=' * 70)
        print(f'\nPSNR Statistics:')
        print(f'  Mean: {np.mean(psnr_list):.2f} dB')
        print(f'  Median: {np.median(psnr_list):.2f} dB')
        print(f'  Std Dev: {np.std(psnr_list):.2f} dB')
        print(f'  Min: {np.min(psnr_list):.2f} dB')
        print(f'  Max: {np.max(psnr_list):.2f} dB')
        print(f'  25th Percentile: {np.percentile(psnr_list, 25):.2f} dB')
        print(f'  75th Percentile: {np.percentile(psnr_list, 75):.2f} dB')

        print(f'\nSSIM Statistics:')
        print(f'  Mean: {np.mean(ssim_list):.4f}')
        print(f'  Median: {np.median(ssim_list):.4f}')
        print(f'  Std Dev: {np.std(ssim_list):.4f}')
        print(f'  Min: {np.min(ssim_list):.4f}')
        print(f'  Max: {np.max(ssim_list):.4f}')
        print(f'  25th Percentile: {np.percentile(ssim_list, 25):.4f}')
        print(f'  75th Percentile: {np.percentile(ssim_list, 75):.4f}')

        # Performance grade statistics
        excellent_psnr = sum(1 for p in psnr_list if p >= 40)
        good_psnr = sum(1 for p in psnr_list if 35 <= p < 40)
        fair_psnr = sum(1 for p in psnr_list if 30 <= p < 35)
        poor_psnr = sum(1 for p in psnr_list if p < 30)

        print(f'\nPerformance Grade Distribution (PSNR):')
        print(f'  Excellent (≥40dB): {excellent_psnr} ({excellent_psnr / len(psnr_list) * 100:.1f}%)')
        print(f'  Good (35-40dB): {good_psnr} ({good_psnr / len(psnr_list) * 100:.1f}%)')
        print(f'  Fair (30-35dB): {fair_psnr} ({fair_psnr / len(psnr_list) * 100:.1f}%)')
        print(f'  Poor (<30dB): {poor_psnr} ({poor_psnr / len(psnr_list) * 100:.1f}%)')

        print('=' * 70)

    def run_analysis(self):
        """Run complete analysis"""
        print('=' * 70)
        print('Complete Model Performance Analysis')
        print('=' * 70)
        print(f'Model: {self.model_path}')
        print(f'Validation Set Size: {len(self.val_dataset)}')
        print(f'Output Directory: {self.output_dir}')
        print('=' * 70)
        print()

        # Generate training curve plot (Figure 0)
        print('Generating analysis plots...')
        print('-' * 70)
        self.plot_0_training_curve()

        # Collect data
        psnr_list, ssim_list, samples = self.collect_metrics()

        # Generate all plots
        self.plot_1_metrics_distribution(psnr_list, ssim_list)
        self.plot_2_sample_comparison(samples, psnr_list)
        self.plot_3_correlation(psnr_list, ssim_list)
        self.plot_4_cumulative_distribution(psnr_list, ssim_list)
        self.plot_5_error_distribution(samples)
        self.plot_6_heatmap(samples)
        self.plot_7_percentile_analysis(psnr_list, ssim_list)
        print('-' * 70)

        # Print statistics
        self.print_statistics(psnr_list, ssim_list)

        print(f'\n✓ All analysis complete! Results saved in: {self.output_dir}')
        print('=' * 70)


if __name__ == '__main__':
    analyzer = ModelAnalyzer(
        model_path='improved_model_best.pth',
        noise_dir='../data/mnist-denoising/data/train/noise',
        origin_dir='../data/mnist-denoising/data/train/origin',
        output_dir='analysis_results',
        loss_history_path='loss_history.json'  # Loss history file path
    )

    analyzer.run_analysis()