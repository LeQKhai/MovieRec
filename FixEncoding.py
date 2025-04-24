import ftfy
import pandas as pd
import re

def fix_encoding(text):
    if pd.isna(text):
        return ""
    fixed_text = ftfy.fix_text(str(text))
    fixed_text = re.sub(r'[^\x00-\x7F\u00C0-\u017F]', '', fixed_text)
    return fixed_text.strip()
