"""
Read and inspect parquet files
"""
import pandas as pd
import argparse
import os

def read_parquet_file(file_path, num_rows=None, show_columns=True):
    """
    Read and display parquet file content
    
    Args:
        file_path: Path to parquet file
        num_rows: Number of rows to display (None = all)
        show_columns: Whether to show column information
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    # Read parquet file
    df = pd.read_parquet(file_path)
    
    print(f"\n{'='*80}")
    print(f"File: {file_path}")
    print(f"{'='*80}")
    
    # Show basic info
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    if show_columns:
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nColumn types:")
        print(df.dtypes)
    
    # Show data
    print(f"\n{'-'*80}")
    if num_rows is not None:
        print(f"First {num_rows} rows:")
        print(df.head(num_rows))
    else:
        print("All data:")
        print(df)
    
    # Show sample of nested data if exists
    print(f"\n{'-'*80}")
    print("Sample record (first row):")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  {df[col].iloc[0]}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and inspect parquet files')
    parser.add_argument('--file', type=str, required=True, help='Path to parquet file')
    parser.add_argument('--num_rows', type=int, default=5, help='Number of rows to display (default: 5, use -1 for all)')
    parser.add_argument('--no_columns', action='store_true', help='Do not show column information')
    
    args = parser.parse_args()
    
    num_rows = None if args.num_rows == -1 else args.num_rows
    
    read_parquet_file(
        file_path=args.file,
        num_rows=num_rows,
        show_columns=not args.no_columns
    )
