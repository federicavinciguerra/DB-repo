"""Databricks Python script entrypoint.

Recommended Databricks Git-provider path: src/train.py
"""

from mlflow_job_demo.train import main


if __name__ == "__main__":
    main()
