import re, os, time, random, requests, pandas as pd
from datetime import datetime, timezone, timedelta

TOTAL_TARGET = 1000
DATE_YEARS_BACK = 3
MIN_CHAIRS = 80
DELAY_MIN = 2.0
DELAY_MAX = 4.0
OUTPUT_DIR = "scarped_data"
OUTPUT_FILE ="primary_reddit_1000.csv"

# Split target evenly across 4 subreddits
PER_SUB = TOTAL_TARGET // 4     # = 250 each
