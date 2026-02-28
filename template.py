import os
from pathlib import Path

# Define the folder and file structure
project_structure = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "etl/extract.py",
    "etl/transform.py",
    "etl/pipeline.py",
    "database/flight_db.duckdb",
    "mlops/train.py",
    "mlops/evaluate.py",
    "mlops/mlflow_tracking.py",
    "api/main.py",
    "frontend/app.py",
    "models/best_model.pkl",
    "monitoring/drift_check.py",
    ".github/workflows/deploy.yml",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    "docker-compose.yml"
]

def create_structure():
    for filepath in project_structure:
        path = Path(filepath)
        # Create directories if they don't exist
        os.makedirs(path.parent, exist_ok=True)
        
        # Create the file if it doesn't exist
        if not path.exists():
            with open(path, "w") as f:
                # Add a simple header for the main files
                if path.suffix == ".py":
                    f.write(f"# {path.name}\nimport os\n")
                elif path.name == "README.md":
                    f.write("# Flight Delay MLOps Project\n")
            print(f"Created: {filepath}")
        else:
            print(f"Skipped (already exists): {filepath}")

if __name__ == "__main__":
    create_structure()
    print("\n✅ Project structure created successfully!")
