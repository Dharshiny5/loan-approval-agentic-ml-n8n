from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib import colors
import subprocess
import json
import os
import shutil
import uuid

app = FastAPI()

BASE_DIR = "c:/Users/dhars/OneDrive/Loan Approval - n8n"
PYTHON_PATH = "c:/Users/dhars/OneDrive/Loan Approval - n8n/venv/Scripts/python.exe"
SCRIPT_PATH = "c:/Users/dhars/OneDrive/Loan Approval - n8n/loan_ml.py"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    run_id = str(uuid.uuid4())[:8]
    temp_csv = os.path.join(BASE_DIR, f"upload_{run_id}.csv")
    output_dir = os.path.join(BASE_DIR, f"output_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    with open(temp_csv, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = subprocess.run(
        [PYTHON_PATH, SCRIPT_PATH, temp_csv, output_dir],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    output = result.stdout
    start = output.find("JSON_OUTPUT_START")
    end = output.find("JSON_OUTPUT_END")

    if start == -1 or end == -1:
        return {
            "error": "Script failed",
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    json_str = output[start + len("JSON_OUTPUT_START"):end].strip()
    data = json.loads(json_str)

    os.remove(temp_csv)

    return data


@app.post("/generate-report")
async def generate_report():

    results_path = os.path.join(BASE_DIR, "loan_results.json")
    with open(results_path, "r") as f:
        data = json.load(f)

    best_model = data["best_model"]
    portfolio = data["portfolio_summary"]
    features = data["feature_importance"][:3]
    top_approved = data["highlight_cases"]["top_approved"]
    top_rejected = data["highlight_cases"]["top_rejected"]

    report_text = f"""
Loan Portfolio Analysis Report
AI-Powered Credit Risk Assessment

Executive Summary
This report presents the findings of an automated loan portfolio analysis conducted on {portfolio['total_applicants']} applicants using a machine learning pipeline. The best-performing model was {best_model['name']} with a ROC-AUC of {best_model['roc_auc']} and F1 score of {best_model['f1']}.

Portfolio Overview
Total Applicants: {portfolio['total_applicants']}
Approved: {portfolio['total_approved']} ({portfolio['approval_rate_pct']}%)
Rejected: {portfolio['total_rejected']}
Risk Distribution: {portfolio['risk_distribution']}

Key Risk Factors
The top three features driving loan decisions were:
1. {features[0]['feature']} (importance: {round(features[0]['importance'], 4)})
2. {features[1]['feature']} (importance: {round(features[1]['importance'], 4)})
3. {features[2]['feature']} (importance: {round(features[2]['importance'], 4)})

Model Performance
Best Model: {best_model['name']}
ROC-AUC: {best_model['roc_auc']}
F1 Score: {best_model['f1']}
Accuracy: {best_model['accuracy']}

Approved Case Highlights
Loan ID {top_approved[0]['loan_id']}: Approved with {top_approved[0]['approval_probability_pct']}% confidence. Annual income {top_approved[0]['income_annum']}, loan amount {top_approved[0]['loan_amount']}. Estimated eligible amount: {top_approved[0]['estimated_eligible_amount']}.

Loan ID {top_approved[1]['loan_id']}: Approved with {top_approved[1]['approval_probability_pct']}% confidence. Annual income {top_approved[1]['income_annum']}, loan amount {top_approved[1]['loan_amount']}. Estimated eligible amount: {top_approved[1]['estimated_eligible_amount']}.

Rejected Case Highlights
Loan ID {top_rejected[0]['loan_id']}: Rejected with {top_rejected[0]['approval_probability_pct']}% approval probability. Risk tier: {top_rejected[0]['risk_tier']}. Loan-to-income ratio of {top_rejected[0]['loan_to_income_ratio']} indicates high debt burden relative to income.

Loan ID {top_rejected[1]['loan_id']}: Rejected with {top_rejected[1]['approval_probability_pct']}% approval probability. Risk tier: {top_rejected[1]['risk_tier']}. Total assets of {top_rejected[1]['total_assets']} insufficient relative to requested loan amount of {top_rejected[1]['loan_amount']}.

Conclusion
The ML pipeline successfully identified key risk drivers in the loan portfolio. The engineered features, particularly loan-to-income and loan-to-assets ratios, proved highly predictive. This automated analysis enables faster, data-driven credit decisions while maintaining explainability and auditability.
"""

    pdf_path = os.path.join(BASE_DIR, "loan_report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1a3c5e'),
        spaceAfter=12,
        spaceBefore=0
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2e6da4'),
        spaceAfter=6,
        spaceBefore=12
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=16,
        spaceAfter=6
    )

    story = []

    lines = report_text.strip().split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
        if i == 0:
            story.append(Paragraph(line, title_style))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=colors.HexColor('#2e6da4')))
        elif i == 1:
            story.append(Paragraph(line, body_style))
        elif line.endswith('Summary') or line.endswith('Overview') or \
             line.endswith('Factors') or line.endswith('Performance') or \
             line.endswith('Highlights') or line.endswith('Conclusion'):
            story.append(Paragraph(line, heading_style))
        else:
            story.append(Paragraph(line, body_style))

    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*inch))

    chart_model = os.path.join(BASE_DIR, "model_comparison.png")
    chart_feature = os.path.join(BASE_DIR, "feature_importance.png")

    story.append(Paragraph("Model Performance Analysis", heading_style))
    if os.path.exists(chart_model):
        story.append(Image(chart_model, width=6*inch, height=3.5*inch))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Feature Importance Analysis", heading_style))
    if os.path.exists(chart_feature):
        story.append(Image(chart_feature, width=6*inch, height=4*inch))

    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="loan_analysis_report.pdf"
    )


@app.get("/lookup/{loan_id}")
async def lookup_loan(loan_id: int):

    results_path = os.path.join(BASE_DIR, "loan_results.json")
    with open(results_path, "r") as f:
        data = json.load(f)

    predictions_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")
    import pandas as pd
    df = pd.read_csv(predictions_path)
    df.columns = df.columns.str.strip()

    row = df[df['loan_id'] == loan_id]

    if row.empty:
        return {"error": f"Loan ID {loan_id} not found"}

    row = row.iloc[0]

    best_model = data["best_model"]
    portfolio = data["portfolio_summary"]

    all_highlights = (
        data["highlight_cases"]["top_approved"] +
        data["highlight_cases"]["top_rejected"] +
        data["highlight_cases"]["borderline"]
    )

    highlight = next((h for h in all_highlights if h["loan_id"] == loan_id), None)

    result = {
        "loan_id": int(loan_id),
        "applicant_info": {
            "no_of_dependents": int(row["no_of_dependents"]),
            "education": row["education"],
            "self_employed": row["self_employed"],
            "income_annum": int(row["income_annum"]),
            "loan_amount": int(row["loan_amount"]),
            "loan_term": int(row["loan_term"]),
            "cibil_score": int(row["cibil_score"]),
            "residential_assets_value": int(row["residential_assets_value"]),
            "commercial_assets_value": int(row["commercial_assets_value"]),
            "luxury_assets_value": int(row["luxury_assets_value"]),
            "bank_asset_value": int(row["bank_asset_value"])
        },
        "model_used": best_model["name"],
        "highlight_data": highlight,
        "portfolio_context": {
            "avg_cibil_approved": portfolio["avg_cibil_approved"],
            "avg_cibil_rejected": portfolio["avg_cibil_rejected"],
            "overall_approval_rate": portfolio["approval_rate_pct"]
        }
    }

    lookup_path = os.path.join(BASE_DIR, f"lookup_{loan_id}.json")
    with open(lookup_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


@app.get("/save-letter/{loan_id}")
async def save_letter(loan_id: int, text: str):

    letter_path = os.path.join(BASE_DIR, f"letter_{loan_id}.txt")
    with open(letter_path, "w", encoding="utf-8") as f:
        f.write(text)

    return {"status": "saved", "loan_id": loan_id}


@app.get("/generate-letter/{loan_id}")
async def generate_letter(loan_id: int):

    letter_path = os.path.join(BASE_DIR, f"letter_{loan_id}.txt")

    if not os.path.exists(letter_path):
        return {"error": f"Letter for Loan ID {loan_id} not found"}

    with open(letter_path, "r", encoding="utf-8") as f:
        letter_text = f.read()

    pdf_path = os.path.join(BASE_DIR, f"loan_letter_{loan_id}.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'LetterTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a3c5e'),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        'LetterHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2e6da4'),
        spaceAfter=6,
        spaceBefore=10
    )

    body_style = ParagraphStyle(
        'LetterBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=18,
        spaceAfter=8
    )

    approved_style = ParagraphStyle(
        'LetterApproved',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,
        spaceAfter=8,
        textColor=colors.HexColor('#1a7a1a')
    )

    rejected_style = ParagraphStyle(
        'LetterRejected',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,
        spaceAfter=8,
        textColor=colors.HexColor('#cc0000')
    )

    story = []
    story.append(Paragraph(f"Loan Application Decision — ID {loan_id}", title_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor('#2e6da4')))
    story.append(Spacer(1, 0.2*inch))

    lines = letter_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.08*inch))
            continue
        line = line.replace('**', '')
        if 'APPROVED' in line.upper() and 'DECISION' in line.upper():
            story.append(Paragraph(line, approved_style))
        elif 'REJECTED' in line.upper() and 'DECISION' in line.upper():
            story.append(Paragraph(line, rejected_style))
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"loan_decision_{loan_id}.pdf"
    )