import os
import pandas as pd
from pathlib import Path

# Try to import openpyxl for Excel formatting
try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

def format_excel_file(filename):
    """
    Apply pretty formatting to an Excel file:
    - Bold headers with colored background
    - Auto-width columns
    - Frozen header row
    - Text wrapping for long cells
    - Borders
    """
    if not OPENPYXL_AVAILABLE:
        return
    
    try:
        # Load the workbook
        wb = load_workbook(filename)
        ws = wb.active
        
        # Define styles
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        wrap_alignment = Alignment(wrap_text=True, vertical='top')
        
        # Format header row
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = border
        
        # Format data cells
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.border = border
                cell.alignment = wrap_alignment
                # Set font for better readability
                cell.font = Font(size=10)
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            
            for cell in column:
                try:
                    # Get the length of the cell value
                    if cell.value:
                        cell_length = len(str(cell.value))
                        # For wrapped text, consider a reasonable max width
                        if cell_length > 50:
                            cell_length = 50  # Cap at 50 characters for width
                        if cell_length > max_length:
                            max_length = cell_length
                except:
                    pass
            
            # Set column width with some padding
            # Minimum width of 10, maximum of 50
            adjusted_width = min(max(max_length + 2, 10), 50)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        # Freeze the header row
        ws.freeze_panes = 'A2'
        
        # Save the formatted workbook
        wb.save(filename)
        
    except Exception as e:
        print(f"  ⚠ Warning: Could not format Excel file: {e}")

def main():
    """
    Main function to combine all CSV and XLSX files from combined_output folder.
    """
    # Define input/output folder
    combined_folder = "combined_output"
    
    if not os.path.exists(combined_folder):
        print(f"⚠ Error: Folder '{combined_folder}' not found!")
        return
    
    print(f"Combining all files from '{combined_folder}' folder...")
    print(f"{'='*60}\n")
    
    # Get all CSV and XLSX files
    csv_files = sorted([f for f in os.listdir(combined_folder) if f.endswith('.csv')])
    xlsx_files = sorted([f for f in os.listdir(combined_folder) if f.endswith('.xlsx')])
    
    if not csv_files and not xlsx_files:
        print(f"⚠ No CSV or XLSX files found in '{combined_folder}'")
        return
    
    # Combine CSV files
    if csv_files:
        print(f"Found {len(csv_files)} CSV file(s):")
        combined_csv_data = []
        
        for csv_file in csv_files:
            csv_path = os.path.join(combined_folder, csv_file)
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
                combined_csv_data.append(df)
                print(f"  ✓ Loaded {len(df)} rows from {csv_file}")
            except Exception as e:
                print(f"  ⚠ Error reading {csv_file}: {e}")
        
        if combined_csv_data:
            # Combine all DataFrames
            combined_df = pd.concat(combined_csv_data, ignore_index=True)
            print(f"\n  Total CSV rows combined: {len(combined_df)}")
            
            # Save combined CSV
            output_csv = os.path.join(combined_folder, "all_data_combined.csv")
            combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"  ✓ Combined CSV saved: '{output_csv}'")
    
    # Combine XLSX files
    if xlsx_files:
        print(f"\nFound {len(xlsx_files)} XLSX file(s):")
        combined_xlsx_data = []
        
        for xlsx_file in xlsx_files:
            xlsx_path = os.path.join(combined_folder, xlsx_file)
            try:
                df = pd.read_excel(xlsx_path, engine='openpyxl')
                combined_xlsx_data.append(df)
                print(f"  ✓ Loaded {len(df)} rows from {xlsx_file}")
            except Exception as e:
                print(f"  ⚠ Error reading {xlsx_file}: {e}")
        
        if combined_xlsx_data:
            # Combine all DataFrames
            combined_df = pd.concat(combined_xlsx_data, ignore_index=True)
            print(f"\n  Total XLSX rows combined: {len(combined_df)}")
            
            # Save combined XLSX
            output_xlsx = os.path.join(combined_folder, "all_data_combined.xlsx")
            combined_df.to_excel(output_xlsx, index=False, engine='openpyxl')
            
            # Apply formatting
            format_excel_file(output_xlsx)
            print(f"  ✓ Combined XLSX saved: '{output_xlsx}'")
    
    print(f"\n{'='*60}")
    print("All files combined successfully!")
    print(f"Output files saved in: '{combined_folder}'")
    print(f"  - all_data_combined.csv")
    print(f"  - all_data_combined.xlsx")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

