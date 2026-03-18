import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
from config import config
from logger import get_logger
from io import StringIO

logger = get_logger(__name__)

def fee_lookup(new_fees):
    base_path = Path(__file__).parent.parent

    existing_fee_csv = base_path / config["database"]["fee_database"]


    existing_fees = pd.read_csv(existing_fee_csv)
    active_existing_fees = existing_fees[existing_fees["is_deprecated"] == False]
    logger.info(f"Reading Fee Database: {existing_fee_csv}")


    # 2. Define a function to find the best match
    def get_fuzzy_match(fee_name, choices, threshold = 85):
        # extractOne returns (match, score, index)
        match = process.extractOne(fee_name, choices, scorer = fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            return match[0]
        return None

    # 3. Apply matching
    # We map 'old' fee names to the closest 'new' fee names
    choices = active_existing_fees["fee_name"].tolist()

    logger.info(f"Starting Fuzzy Match")
    new_fees["matched_fee_name"] = new_fees["fee_name"].apply(
        lambda x: get_fuzzy_match(x, choices)
    )
    logger.info(f"Database fuzz match result:\n{new_fees}")

    # 4. Merge the tables
    logger.info(f"Mapping rates in fee_database to fee_announcement_database")

    merged_df = pd.merge(
        new_fees, 
        active_existing_fees,
        left_on=["matched_fee_name","region"], 
        right_on=["fee_name","region"], 
        how="left",
        suffixes=("_existing", "_announcement")
    )
    logger.info(f"Dropping Columns... Renaming Columns.... Ordering Column...")
    # merged_df = merged_df.drop(["extracted_at","matched_fee_name","fee_name_announcement","start_date"], axis=1)
    merged_df = merged_df.rename(columns={"fee_name_existing":"fee_name"})
    column_order = ["region", "effective_date", "fee_name","current_rate", "new_rate", "change_type"]
    merged_df = merged_df[column_order]

    updated_fee_markdown = merged_df.to_markdown(index=False)
    logger.info(f"Output\n {updated_fee_markdown}")
    return updated_fee_markdown

def add_fees(new_fees):
    update_new_fees = new_fees[new_fees["change_type"] == "updated_fee"]



if __name__ == "__main__":
    test_data = """
"fee_name","new_rate","effective_date","region","currency","change_type"
"Digital Assurance Acquirer Fee – Non-Tokenized (Debit)","0.04","2025-10-13","Australia","AUD","updated_fee"
"Digital Assurance Acquirer Fee – Non-Tokenized (Credit)","0.04","2025-10-13","Australia","AUD","updated_fee"
    """
    test_data_df = pd.read_csv(StringIO(test_data))
    # print(test_data_df)

    fee_lookup(new_fees = test_data_df)





