import pandas as pd
from ydata_profiling import ProfileReport


def generate_profile(file_path: str, output_path: str) -> None:
    df = pd.read_csv(file_path)

    profile = ProfileReport(
        df,
        title="Dataset Profile",
    )

    profile.to_file(output_path)