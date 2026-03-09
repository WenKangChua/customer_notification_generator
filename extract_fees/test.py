import json 


eg = """
{
  "fee_names": [
    {
      "fee_name": "Digital Assurance Acquirer Fee – Non-Tokenized (Debit)",
      "new_rate": 0.04,
      "effective_date": "2025-10-13",
      "country": "Australia",
      "change_type": "updated_fee"
    },
    {
      "fee_name": "Digital Assurance Acquirer Fee – Non-Tokenized (Credit)",
      "new_rate": 0.04,
      "effective_date": "2025-10-13",
      "country": "Australia",
      "change_type": "updated_fee"
    }
  ]
}
"""

print(json.loads(eg)["fee_names"])