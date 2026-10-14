import os
import pandas as pd
from fpdf import FPDF # pip install fpdf

os.makedirs("data/pdfs", exist_ok=True)

# 1. Generate Synthetic Mortgage PDFs
clauses = [
    ("LN-40291", "Clause 4.2", "Max margin increase 0.50% in 12 months with 30 days notice."),
    ("LN-50112", "Clause 7.1", "Early repayment charge is 2% of the outstanding balance."),
    ("LN-66231", "Clause 3.3", "Interest calculated daily and applied monthly."),
]

for loan, clause, text in clauses:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Mortgage Agreement: {loan}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"{clause} - {text}", ln=True)
    pdf.output(f"data/pdfs/{loan}.pdf")

# 2. Generate CSV for ML Router Training
data = {
    "query": [
        "What is my current balance?", 
        "Change my home address", 
        "Did the bank breach Clause 4.2 by raising rates 0.75%?", 
        "Is the early repayment charge applicable if I sell?"
    ],
    "intent": ["simple", "simple", "complex", "complex"]
}
pd.DataFrame(data).to_csv("data/historical_queries.csv", index=False)
print("Data generation complete!")