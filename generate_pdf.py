from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Executive Summary: Credit Risk ML Model', new_x="RIGHT", new_y="TOP", align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', new_x="RIGHT", new_y="TOP", align='C')

pdf = PDF()
pdf.add_page()

pdf.set_font("helvetica", 'B', 12)
pdf.cell(190, 10, text="1. The Business Challenge", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", size=11)
pdf.multi_cell(190, 6, text="Traditional models optimise for accuracy and apply a 0.5 decision threshold, assuming all errors are equally costly. In credit risk, that assumption is wrong by an order of magnitude. Missing a default costs GBP 25,000 (principal loss + collections), while wrongly rejecting a good customer costs GBP 2,500 (lost interest). This 10:1 cost ratio requires a specialised, cost-sensitive approach.")

pdf.ln(5)
pdf.set_font("helvetica", 'B', 12)
pdf.cell(190, 10, text="2. Business Result & Financial Impact", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", size=11)
pdf.multi_cell(190, 6, text="By replacing the naive 0.5 probability threshold with a mathematically derived optimal threshold of 0.23, the XGBoost model transformed abstract ML accuracy into a quantifiable financial outcome:")
pdf.ln(2)
pdf.set_font("helvetica", 'B', 11)
pdf.multi_cell(190, 6, text="Impact on the 6,000-customer test portfolio:")
pdf.set_font("helvetica", size=11)
pdf.multi_cell(190, 6, text="- Defaults caught: Increased from 741 to 1,281 (+540 additional defaults).")
pdf.multi_cell(190, 6, text="- Detection rate: Improved by 72.8 percent.")
pdf.multi_cell(190, 6, text="- Net financial saving: GBP 5,960,000 compared to the baseline.")
pdf.ln(2)
pdf.multi_cell(190, 6, text="The GBP 5.96M figure accounts for every false positive incurred at the lower threshold - the saving is real net of intervention cost.")

pdf.ln(5)
pdf.set_font("helvetica", 'B', 12)
pdf.cell(190, 10, text="3. Performance & Threshold Analysis", new_x="LMARGIN", new_y="NEXT")
try:
    pdf.image('outputs/figures/cost_curve.png', x=10, y=None, w=90)
    y_current = pdf.get_y()
    pdf.image('outputs/figures/confusion_matrices.png', x=10, y=y_current + 5, w=180)
except Exception as e:
    pdf.cell(190, 10, text=f"[Could not load images: {e}]", new_x="LMARGIN", new_y="NEXT")

pdf.add_page()

pdf.set_font("helvetica", 'B', 12)
pdf.cell(190, 10, text="4. Explainability and Risk Factors", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", size=11)
pdf.multi_cell(190, 6, text="The model utilizes SHAP (SHapley Additive exPlanations) to provide full transparency. The beeswarm plot below shows the most critical risk drivers across the entire portfolio. Payment delays (PAY_1), low limit balances, and high utilization rates are the strongest indicators of default.")

try:
    pdf.image('outputs/figures/shap_beeswarm.png', x=30, y=None, w=150)
except Exception as e:
    pass

pdf.ln(10)
pdf.set_font("helvetica", 'B', 12)
pdf.cell(190, 10, text="5. Conclusion & Recommendations", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("helvetica", size=11)
pdf.multi_cell(190, 6, text="The cost-sensitive model successfully bridges the gap between technical metrics and business ROI. We recommend deploying this model in a shadow-testing environment to validate the projected GBP 5.96M savings per 6,000 applicants, while actively monitoring the top SHAP features for macroeconomic drift.")

pdf.output('outputs/executive_summary.pdf')
print("PDF generated successfully.")
