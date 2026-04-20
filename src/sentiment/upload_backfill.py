"""One-shot bulk upload after the 3-4 day backfill run. Run manually once."""
import glob
import logging

import pandas as pd

from Utilities import Database as DB

log = logging.getLogger(__name__)


def upload_backfill(parquet_root: str = "parquet_backfill") -> None:
    for table, folder, id_col in [
        ("sentiment_events",   "event",   "docid"),
        ("sentiment_speakers", "speaker", "docid_speakerid"),
        ("sentiment_segments", "segment", "docid_paragraphindex"),
    ]:
        files = sorted(glob.glob(f"{parquet_root}/{folder}/batch_*.parquet"))
        if not files:
            log.warning("no parquet files found for %s", folder)
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.drop_duplicates(subset=[id_col], keep="last")
        log.info("uploading table=%s rows=%d", table, len(df))
        DB.upload_msql(df=df, schema="raw", table=table, db="EDI", replace=0, id_col_name=id_col)


if __name__ == "__main__":
    upload_backfill()
