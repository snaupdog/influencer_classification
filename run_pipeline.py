"""
Automated Pipeline Runner
Executes all steps of the influencer classification pipeline in sequence.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


class PipelineRunner:
    def __init__(self, root_dir=None):
        """Initialize the pipeline runner."""
        if root_dir is None:
            root_dir = os.getcwd()
        self.root_dir = Path(root_dir)
        self.start_time = time.time()
        self.steps_completed = []
        self.steps_failed = []

    def log(self, message, level="INFO"):
        """Print formatted log message."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if level == "ERROR":
            print(f"❌ [{timestamp}] {message}")
        elif level == "SUCCESS":
            print(f"✅ [{timestamp}] {message}")
        elif level == "WARNING":
            print(f"⚠️  [{timestamp}] {message}")
        else:
            print(f"🔹 [{timestamp}] {message}")

    def run_command(self, cmd, cwd=None, description=""):
        """Execute a shell command and handle errors."""
        if cwd is None:
            cwd = self.root_dir

        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"Working directory: {cwd}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=False,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode == 0:
                self.log(f"✓ {description} completed successfully", "SUCCESS")
                return True
            else:
                self.log(
                    f"✗ {description} failed with exit code {result.returncode}",
                    "ERROR",
                )
                return False
        except subprocess.TimeoutExpired:
            self.log(f"✗ {description} timed out after 1 hour", "ERROR")
            return False
        except Exception as e:
            self.log(f"✗ Error running {description}: {str(e)}", "ERROR")
            return False

    def check_requirements(self):
        """Check if all required directories and files exist."""
        self.log("Checking requirements...", "INFO")

        required_dirs = [
            "gen_dataset",
            "dataset",
            "image_embed",
            "text_embed",
            "combined_features",
            "classifier",
        ]

        required_files = {
            "gen_dataset/influencers_17.csv": "Influencers data file",
            "gen_dataset/JSON-image_17.csv": "JSON-image mapping file",
            "image_embed/image_feature_extractor.keras": "Image feature extractor model",
            "classifier/influencer_profiler_best.keras": "Classification model",
        }

        missing = False

        for dir_name in required_dirs:
            dir_path = self.root_dir / dir_name
            if not dir_path.exists():
                self.log(f"Missing directory: {dir_name}", "WARNING")
                missing = True

        for file_path, description in required_files.items():
            full_path = self.root_dir / file_path
            if not full_path.exists():
                self.log(f"Missing file: {file_path} ({description})", "WARNING")
                missing = True

        if missing:
            self.log("Some files are missing. Pipeline may fail.", "WARNING")
            response = input("Continue anyway? (y/n): ")
            return response.lower() == "y"

        self.log("All requirements checked", "SUCCESS")
        return True

    def step_1_generate_influencer_metadata(self, n_influencers=10, n_posts=5):
        """Step 1: Generate influencer metadata."""
        self.log("\n" + "=" * 70)
        self.log("STEP 1: Generate Influencer Metadata", "INFO")
        self.log("=" * 70)

        script_path = self.root_dir / "gen_dataset" / "generating_influencer_metdata.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        # Create a wrapper script to auto-input values
        wrapper_script = f"""
import subprocess
import sys

process = subprocess.Popen(
    [sys.executable, 'generating_influencer_metdata.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate(input="{n_influencers}\\n{n_posts}\\n")
print(stdout)
if stderr:
    print(stderr, file=sys.stderr)

sys.exit(process.returncode)
"""

        wrapper_path = self.root_dir / "gen_dataset" / "_wrapper_influencer.py"
        with open(wrapper_path, "w") as f:
            f.write(wrapper_script)

        success = self.run_command(
            [sys.executable, "_wrapper_influencer.py"],
            cwd=self.root_dir / "gen_dataset",
            description="Generate Influencer Metadata",
        )

        # Cleanup wrapper
        wrapper_path.unlink()

        if success:
            self.steps_completed.append("Step 1: Generate Influencer Metadata")
        else:
            self.steps_failed.append("Step 1: Generate Influencer Metadata")

        return success

    def step_2_generate_images_info(self):
        """Step 2: Generate dataset images info."""
        self.log("\n" + "=" * 70)
        self.log("STEP 2: Generate Dataset Images Info", "INFO")
        self.log("=" * 70)

        script_path = self.root_dir / "dataset" / "generating_images_info.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        success = self.run_command(
            [sys.executable, "generating_images_info.py"],
            cwd=self.root_dir / "dataset",
            description="Generate Images Info",
        )

        if success:
            self.steps_completed.append("Step 2: Generate Dataset Images Info")
        else:
            self.steps_failed.append("Step 2: Generate Dataset Images Info")

        return success

    def step_3_compress_preprocess(self):
        """Step 3: Compress and preprocess images."""
        self.log("\n" + "=" * 70)
        self.log("STEP 3: Compress and Preprocess Images", "INFO")
        self.log("=" * 70)

        script_path = self.root_dir / "image_embed" / "compressPreprocess.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        success = self.run_command(
            [sys.executable, "compressPreprocess.py"],
            cwd=self.root_dir / "image_embed",
            description="Compress and Preprocess Images",
        )

        if success:
            self.steps_completed.append("Step 3: Compress and Preprocess Images")
        else:
            self.steps_failed.append("Step 3: Compress and Preprocess Images")

        return success

    def step_4_extract_image_features(self):
        """Step 4: Extract image features."""
        self.log("\n" + "=" * 70)
        self.log("STEP 4: Extract Image Features", "INFO")
        self.log("=" * 70)

        script_path = self.root_dir / "image_embed" / "extract_image_features.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        success = self.run_command(
            [sys.executable, "extract_image_features.py"],
            cwd=self.root_dir / "image_embed",
            description="Extract Image Features",
        )

        if success:
            self.steps_completed.append("Step 4: Extract Image Features")
        else:
            self.steps_failed.append("Step 4: Extract Image Features")

        return success

    def step_5_process_text(self):
        """Step 5: Process text and generate initial dataset."""
        self.log("\n" + "=" * 70)
        self.log("STEP 5: Process Text Pipeline", "INFO")
        self.log("=" * 70)

        script_path = self.root_dir / "text_embed" / "pipeline.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        success = self.run_command(
            [sys.executable, "pipeline.py"],
            cwd=self.root_dir / "text_embed",
            description="Process Text Pipeline",
        )

        if success:
            self.steps_completed.append("Step 5: Process Text Pipeline")
        else:
            self.steps_failed.append("Step 5: Process Text Pipeline")

        return success

    def step_6_embed_text(self):
        """Step 6: Embed text with BERT."""
        self.log("\n" + "=" * 70)
        self.log("STEP 6: Embed Text with BERT", "INFO")
        self.log("=" * 70)

        script_path = self.root_dir / "text_embed" / "embed.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        success = self.run_command(
            [sys.executable, "embed.py"],
            cwd=self.root_dir / "text_embed",
            description="Embed Text with BERT",
        )

        if success:
            self.steps_completed.append("Step 6: Embed Text with BERT")
        else:
            self.steps_failed.append("Step 6: Embed Text with BERT")

        return success

    def step_7_combine_features(self):
        """Step 7: Combine image and text features."""
        self.log("\n" + "=" * 70)
        self.log("STEP 7: Combine Feature Vectors", "INFO")
        self.log("=" * 70)

        script_path = (
            self.root_dir / "combined_features" / "combined_feature_vectors.py"
        )
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        success = self.run_command(
            [sys.executable, "combined_feature_vectors.py"],
            cwd=self.root_dir / "combined_features",
            description="Combine Feature Vectors",
        )

        if success:
            self.steps_completed.append("Step 7: Combine Feature Vectors")
        else:
            self.steps_failed.append("Step 7: Combine Feature Vectors")

        return success

    def step_8_run_classifier(self):
        """Step 8: Run classification."""
        self.log("\n" + "=" * 70)
        self.log("STEP 8: Run Classification", "INFO")
        self.log("=" * 70)

        script_path = self.root_dir / "classifier" / "run_classificatoin_for_folder.py"
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False

        success = self.run_command(
            [sys.executable, "run_classificatoin_for_folder.py"],
            cwd=self.root_dir / "classifier",
            description="Run Classification",
        )

        if success:
            self.steps_completed.append("Step 8: Run Classification")
        else:
            self.steps_failed.append("Step 8: Run Classification")

        return success

    def run_pipeline(self, n_influencers=10, n_posts=5, skip_steps=None):
        """Run the complete pipeline."""
        if skip_steps is None:
            skip_steps = []

        self.log("\n" + "=" * 70)
        self.log("🚀 STARTING INFLUENCER CLASSIFICATION PIPELINE", "INFO")
        self.log("=" * 70)
        self.log(f"Root directory: {self.root_dir}\n")

        # Check requirements
        if not self.check_requirements():
            self.log("Requirements check failed. Exiting.", "ERROR")
            return False

        steps = [
            (1, self.step_1_generate_influencer_metadata, [n_influencers, n_posts]),
            (2, self.step_2_generate_images_info, []),
            (3, self.step_3_compress_preprocess, []),
            (4, self.step_4_extract_image_features, []),
            (5, self.step_5_process_text, []),
            (6, self.step_6_embed_text, []),
            (7, self.step_7_combine_features, []),
            (8, self.step_8_run_classifier, []),
        ]

        for step_num, step_func, args in steps:
            if step_num in skip_steps:
                self.log(f"Skipping Step {step_num} as requested", "WARNING")
                continue

            success = step_func(*args)
            if not success:
                response = input(
                    f"\nStep {step_num} failed. Continue with remaining steps? (y/n): "
                )
                if response.lower() != "y":
                    self.log("Pipeline execution stopped by user", "WARNING")
                    break

        self.print_summary()
        return len(self.steps_failed) == 0

    def print_summary(self):
        """Print pipeline execution summary."""
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        self.log("\n" + "=" * 70)
        self.log("PIPELINE EXECUTION SUMMARY", "INFO")
        self.log("=" * 70)

        self.log(f"Total execution time: {hours}h {minutes}m {seconds}s", "INFO")
        self.log(
            f"Steps completed: {len(self.steps_completed)}/{len(self.steps_completed) + len(self.steps_failed)}"
        )

        if self.steps_completed:
            self.log("\n✅ Completed steps:", "SUCCESS")
            for step in self.steps_completed:
                print(f"   {step}")

        if self.steps_failed:
            self.log("\n❌ Failed steps:", "ERROR")
            for step in self.steps_failed:
                print(f"   {step}")

        if not self.steps_failed:
            self.log("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!", "SUCCESS")
        else:
            self.log(f"\n⚠️  {len(self.steps_failed)} step(s) failed", "WARNING")

        self.log("=" * 70 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Influencer Classification Pipeline Runner"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.getcwd(),
        help="Root directory of the project (default: current directory)",
    )
    parser.add_argument(
        "--influencers",
        type=int,
        default=10,
        help="Number of influencers per category (default: 10)",
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=5,
        help="Number of posts per influencer (default: 5)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="+",
        help="Step numbers to skip (e.g., --skip 1 3 will skip steps 1 and 3)",
    )

    args = parser.parse_args()

    runner = PipelineRunner(root_dir=args.root)
    success = runner.run_pipeline(
        n_influencers=args.influencers,
        n_posts=args.posts,
        skip_steps=args.skip or [],
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
