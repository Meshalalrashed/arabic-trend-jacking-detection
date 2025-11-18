import asyncio
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from twscrape import API, Tweet

# Try to import openpyxl for Excel formatting
try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

def extract_all_attributes(obj, prefix=""):
    """
    Recursively extract all attributes from an object.
    Handles nested objects, lists, and complex types.
    """
    data = {}
    
    if obj is None:
        return None
    
    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle lists
    if isinstance(obj, list):
        return [extract_all_attributes(item, prefix) for item in obj]
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: extract_all_attributes(v, prefix) for k, v in obj.items()}
    
    # Extract all attributes from the object
    try:
        for attr_name in dir(obj):
            # Skip private/magic methods and methods
            if attr_name.startswith('_') or callable(getattr(obj, attr_name, None)):
                continue
            
            try:
                attr_value = getattr(obj, attr_name, None)
                
                # Skip None values or methods
                if attr_value is None or callable(attr_value):
                    continue
                
                # Create the key name
                key = f"{prefix}_{attr_name}" if prefix else attr_name
                
                # Recursively extract nested objects
                if hasattr(attr_value, '__dict__') or hasattr(attr_value, '__class__'):
                    # Check if it's a simple object we can extract from
                    try:
                        data[key] = extract_all_attributes(attr_value, key)
                    except:
                        # If extraction fails, convert to string
                        data[key] = str(attr_value)
                else:
                    data[key] = extract_all_attributes(attr_value, key)
                    
            except Exception as e:
                # Skip attributes that can't be accessed
                continue
                
    except Exception as e:
        # If we can't extract attributes, return string representation
        return str(obj)
    
    return data

def sanitize_filename(query):
    """Convert query to a safe filename."""
    # Replace special characters and spaces with underscores
    filename = query.replace("#", "").replace(" ", "_").replace("(", "").replace(")", "")
    # Remove or replace Arabic characters for filename compatibility
    filename = filename.replace("موسم_الرياض", "riyadh_season_ar")
    return filename

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

async def scrape_query(api, query, tweet_limit, query_index, since_date=None, until_date=None, output_folder=None):
    """
    Scrape tweets for a single query and save to CSV and XLSX files.
    If since_date and until_date are provided, filters tweets by date range.
    Files are saved in the specified output_folder.
    """
    # Build the search query with date filters if provided
    search_query = query
    if since_date and until_date:
        # Twitter search syntax: since:YYYY-MM-DD until:YYYY-MM-DD
        search_query = f"{query} since:{since_date} until:{until_date}"
        print(f"\n{'='*60}")
        print(f"Query {query_index}: {query}")
        print(f"Date range: {since_date} to {until_date}")
        print(f"Will scrape a maximum of {tweet_limit} tweets...")
    else:
        print(f"\n{'='*60}")
        print(f"Query {query_index}: {query}")
        print(f"Will scrape a maximum of {tweet_limit} tweets...")
    print(f"{'='*60}\n")

    # This list will hold all our tweet data for this query
    all_tweets_data = []

    try:
        # api.search() returns a generator. We loop through it.
        async for tweet in api.search(search_query, limit=tweet_limit):
            if tweet:
                # Extract ALL available information from the tweet object
                tweet_data = extract_all_attributes(tweet, "tweet")
                
                # Also extract all user information if available
                if tweet.user:
                    user_data = extract_all_attributes(tweet.user, "user")
                    # Merge user data into tweet data
                    tweet_data.update(user_data)
                
                # Flatten nested dictionaries for better CSV compatibility
                # Convert nested dicts to JSON strings for CSV storage
                flattened_data = {}
                for key, value in tweet_data.items():
                    if isinstance(value, dict):
                        flattened_data[key] = json.dumps(value, default=str, ensure_ascii=False)
                    elif isinstance(value, list):
                        # Convert lists to JSON strings, or comma-separated if simple
                        if value and isinstance(value[0], (str, int, float, bool)):
                            flattened_data[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            flattened_data[key] = json.dumps(value, default=str, ensure_ascii=False)
                    else:
                        flattened_data[key] = value
                
                all_tweets_data.append(flattened_data)

                # Print progress every 100 tweets
                if len(all_tweets_data) % 100 == 0:
                    print(f"  Scraped {len(all_tweets_data)} tweets for query '{query}'...")

    except Exception as e:
        print(f"  An error occurred while scraping '{query}': {e}")
    finally:
        # This part runs whether the scrape finished or failed,
        # ensuring we save whatever data we got.
        if all_tweets_data:
            print(f"\n  Scraping finished for '{query}'. Total tweets collected: {len(all_tweets_data)}")
            
            # Convert the list of dictionaries into a pandas DataFrame
            df = pd.DataFrame(all_tweets_data)
            
            # Create safe filename from query and date if provided
            base_filename = sanitize_filename(query)
            if since_date:
                # Add date to filename
                base_filename = f"{base_filename}_{since_date}"
            
            # Use output_folder if provided, otherwise save in current directory
            if output_folder:
                # Ensure the folder exists
                os.makedirs(output_folder, exist_ok=True)
                csv_filename = os.path.join(output_folder, f"{base_filename}.csv")
                xlsx_filename = os.path.join(output_folder, f"{base_filename}.xlsx")
            else:
                csv_filename = f"{base_filename}.csv"
                xlsx_filename = f"{base_filename}.xlsx"
            
            # Save the DataFrame to a CSV file
            # 'encoding="utf-8-sig"' is important for saving Arabic text correctly
            df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
            print(f"  ✓ Data saved to CSV: '{csv_filename}'")
            
            # Save the DataFrame to an XLSX file with pretty formatting
            # openpyxl engine handles Unicode better for Excel files
            try:
                df.to_excel(xlsx_filename, index=False, engine='openpyxl')
                # Apply pretty formatting
                format_excel_file(xlsx_filename)
                print(f"  ✓ Data saved to XLSX: '{xlsx_filename}' (formatted)")
            except ImportError:
                print(f"  ⚠ Warning: openpyxl not installed. Install it with 'pip install openpyxl' to save XLSX files.")
                print(f"  ⚠ Skipping XLSX save for '{query}'")
            except Exception as e:
                print(f"  ⚠ Error saving XLSX file: {e}")
        else:
            print(f"  No tweets were collected for query '{query}'.")

# This is the main function where all the scraping happens
async def main():
    # Initialize API (default is accounts.db)
    api = API()
    
    # Load accounts from account.txt file
    # Format: username:pass:email:email_pass:cookies_string (one account per line)
    # cookies_string format: "ct0=value; auth_token=value; twid=value" or '{"ct0": "value", "auth_token": "value"}'
    accounts_file = "account.txt"
    
    accounts_added = 0
    accounts_failed = 0
    
    try:
        with open(accounts_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if not lines:
            print(f"⚠ Warning: '{accounts_file}' is empty")
        else:
            print(f"Reading accounts from '{accounts_file}'...")
            
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse the line: username:pass:email:email_pass:cookies_string
                # Split by ':' but only take first 4 parts, rest is cookies (may contain ':')
                parts = line.split(':', 4)  # Split into max 5 parts
                if len(parts) >= 5:
                    username = parts[0].strip()
                    password = parts[1].strip()
                    email = parts[2].strip()
                    email_password = parts[3].strip()
                    cookies = parts[4].strip()
                    
                    if not all([username, password, email, email_password, cookies]):
                        print(f"  ⚠ Line {line_num}: Missing required fields, skipping")
                        accounts_failed += 1
                        continue
                    
                    try:
                        # Delete account if it already exists (to update it)
                        try:
                            await api.pool.delete_accounts(username)
                            print(f"  Removed existing account '{username}' to update it")
                        except:
                            pass
                        
                        # Option 1: Adding account with cookies (more stable)
                        # cookies can be: "abc=12; ct0=xyz" or '{"abc": "12", "ct0": "xyz"}'
                        await api.pool.add_account(username, password, email, email_password, cookies=cookies)
                        print(f"  ✓ Account '{username}' added with cookies")
                        accounts_added += 1
                        
                    except Exception as e:
                        print(f"  ⚠ Failed to add account '{username}': {e}")
                        accounts_failed += 1
                else:
                    print(f"  ⚠ Line {line_num}: Invalid format. Expected: username:pass:email:email_pass:cookies_string")
                    accounts_failed += 1
            
            print(f"\n✓ Accounts added: {accounts_added}")
            if accounts_failed > 0:
                print(f"⚠ Accounts failed: {accounts_failed}")
            
            # Optionally try to login to refresh session (may fail due to Cloudflare)
            # Uncomment the next line if you want to try login_all()
            # await api.pool.login_all()
                    
    except FileNotFoundError:
        print(f"⚠ Warning: '{accounts_file}' not found. Using existing accounts.db")
    except Exception as e:
        print(f"⚠ Error reading '{accounts_file}': {e}")
        print("⚠ Using existing accounts.db")
    
    # Define your search queries separately (one by one)
    queries = [
        "#جيش_البرهان_يرفض_الهدنه"
    ]
    
    # Define the total number of tweets you want per day
    tweet_limit = 5000
    
    # Define start date and number of days to go back
    start_date_str = "2025-11-07"  # Format: YYYY-MM-DD
    num_days = 7  # Number of days to scrape (going back from start_date)
    
    # Parse start date
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    
    print(f"Starting scrape for {len(queries)} queries")
    print(f"Start date: {start_date_str}")
    print(f"Number of days: {num_days}")
    print(f"Each day will scrape a maximum of {tweet_limit} tweets\n")

    # Process each query, day by day
    for query_index, query in enumerate(queries, start=1):
        print(f"\n{'#'*60}")
        print(f"Processing Query {query_index}: {query}")
        print(f"{'#'*60}")
        
        # Create folder for this query
        query_folder = sanitize_filename(query)
        os.makedirs(query_folder, exist_ok=True)
        print(f"  Created folder: '{query_folder}'")
        
        # Loop through each day, starting from start_date and going back
        for day_offset in range(num_days):
            # Calculate the date for this day (going back from start_date)
            current_date = start_date - timedelta(days=day_offset)
            next_date = current_date + timedelta(days=1)
            
            # Format dates as YYYY-MM-DD strings
            since_date_str = current_date.strftime("%Y-%m-%d")
            until_date_str = next_date.strftime("%Y-%m-%d")
            
            print(f"\n{'='*60}")
            print(f"Day {day_offset + 1}/{num_days}: Scraping tweets from {since_date_str}")
            print(f"{'='*60}")
            
            # Scrape this day and save to query folder
            await scrape_query(
                api, 
                query, 
                tweet_limit, 
                query_index, 
                since_date=since_date_str, 
                until_date=until_date_str,
                output_folder=query_folder
            )
    
    print(f"\n{'='*60}")
    print("All queries completed!")
    print(f"{'='*60}")

# This is the standard way to run an "async" Python script
if __name__ == "__main__":
    asyncio.run(main())