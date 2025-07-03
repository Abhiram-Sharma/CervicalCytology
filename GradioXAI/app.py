import gradio as gr
import datetime
from PIL import Image
import tempfile
import cv2
import os
import numpy as np
from jinja2 import Template
from pipeline.evaluator import evaluate_image
import base64
import webbrowser

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "pipeline", "models", "densenet_trained_model_recovery.h5")
STARDIST_MODEL_DIR = os.path.join(BASE_DIR, "stardist model", "python_2D_versatile_fluo")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def safe_text(text):
    return str(text).encode("utf-8", "ignore").decode("utf-8", "ignore")

def analyze_slide(patient_id, slide_id, image):
    if not patient_id or not slide_id or image is None:
        return ("", "", "", "", None, "❌ Please provide all fields.", "", "Please fill all required fields.", None, None)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        temp_path = temp.name

    report_dir = os.path.join(REPORTS_DIR, f"report_{patient_id}_{slide_id}")
    os.makedirs(report_dir, exist_ok=True)

    original_img_path = os.path.join(report_dir, "original_input.png")
    overlay_img_path = os.path.join(report_dir, "final_overlay_output.png")

    try:
        overlay_img, results = evaluate_image(temp_path, MODEL_WEIGHTS_PATH, STARDIST_MODEL_DIR)

        if isinstance(overlay_img, np.ndarray):
            overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            overlay_img_pil = Image.fromarray(overlay_img_rgb)
            overlay_img_pil.save(overlay_img_path, format='PNG', optimize=True)
        else:
            overlay_img.save(overlay_img_path, format='PNG', optimize=True)

        image.save(original_img_path, format='PNG', optimize=True)

    finally:
        os.remove(temp_path)

    if results is None:
        prediction = "Normal"
        alert_msg = "✅ No abnormality detected."
        cell_distribution = {"low_grade": 0, "high_grade": 0, "cancer": 0}
        confidence = 1.0
    else:
        prediction_raw = results["majority_class"]
        confidence = results.get("confidence", None)
        prediction = prediction_raw.replace("_", " ").capitalize()
        if confidence is not None:
            prediction += f" ({confidence * 100:.2f}%)"
        alert_msg = safe_text(results["alert_message"])
        cell_distribution = results["class_counts"]

    date_today = datetime.datetime.now().strftime("%d %b %Y")

    from LLM import LLM_Analysis
    result = LLM_Analysis(prediction, overlay_img_path)

    summary = (
        f"Analysis Summary:\n{result}" if "Normal" not in prediction else
        "Analysis Summary:\n- No abnormal cells detected.\n- Routine follow-up advised."
    )

    classwise_output = (
        f"Low Grade Cells: {cell_distribution.get('low_grade', 0)}\n"
        f"High Grade Cells: {cell_distribution.get('high_grade', 0)}\n"
        f"Cancerous Cells: {cell_distribution.get('cancer', 0)}"
    )

    overlay_image_to_show = overlay_img_pil if isinstance(overlay_img, np.ndarray) else overlay_img

    return (
        date_today,
        patient_id,
        slide_id,
        prediction,
        overlay_image_to_show,
        alert_msg,
        classwise_output,
        summary,
        original_img_path,
        overlay_img_path
    )

def encode_image_base64(image_path):
    if not os.path.isfile(image_path):
        print(f"Warning: Image file not found at {image_path}")
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_html_report(date, patient_id, slide_id, prediction, alert, cell_dist, summary,
                         original_img_path, overlay_img_path, remarks):
    test_datetime = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    prediction_clean = prediction.split("(")[0].strip().lower()

    status_class = {
        "normal": "status-normal",
        "low grade": "status-abnormal",
        "high grade": "status-abnormal",
        "cancer": "status-malignant"
    }
    prediction_class = status_class.get(prediction_clean, "status-inconclusive")

    table_rows = ""
    total_cells = 0
    counts = {}

    for line in cell_dist.strip().split("\n"):
        if ":" in line:
            label, count = line.split(":")
            label = label.strip().replace("_", " ").capitalize()
            try:
                count = int(count.strip())
            except ValueError:
                count = 0
            counts[label] = count
            total_cells += count

    for label, count in counts.items():
        percentage = f"{(count / total_cells) * 100:.2f}%" if total_cells > 0 else "—"
        table_rows += f"<tr><td>{label}</td><td>{count}</td><td>{percentage}</td></tr>\n"

    alert_block = ""
    if "⚠️" in alert or "❌" in alert:
        alert_block = f"""
        <div class="alert-message">
            <span class="alert-title">⚠️ Alert: Review Recommended</span>
            <p>{alert}</p>
        </div>
        """

    template_path = os.path.join(BASE_DIR, "custom_report_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    html = template.render(
        patient_name="ABC DEF",
        patient_id=patient_id,
        patient_dob="14/08/2005",
        patient_gender="Female",
        slide_id=slide_id,
        test_datetime=test_datetime,
        original_image_src="data:image/png;base64," + encode_image_base64(original_img_path),
        overlaid_image_src="data:image/png;base64," + encode_image_base64(overlay_img_path),
        prediction_class=prediction_class,
        overall_prediction=prediction,
        cell_distribution_table_rows=table_rows,
        qualitative_analysis_text=summary.replace("\n", "<br>"),
        model_confidence_score=prediction.split("(")[-1].replace(")", "") if "(" in prediction else "—",
        alert_message_html_block=alert_block,
        pathologist_remarks=remarks,
        reporting_pathologist_name="XYZ",
        report_generation_datetime=test_datetime,
        lab_name="AI Pathology Lab",
        lab_address="Bangalore, Tamil Nadu",
        lab_phone="+91-93606-45949"
    )

    report_path = os.path.join(os.path.dirname(original_img_path), "custom_report_template_rendered.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open('file://' + os.path.abspath(report_path))
    return report_path


# === Custom CSS for Buttons ===
custom_css = """
button.analyze-btn, button.report-btn {
    background-color: #1565C0;
    color: white;
    font-weight: bold;
    font-size: 15px;
    border-radius: 6px;
    border: none;
    padding: 10px 20px;
    transition: background-color 0.3s ease;
}
button.analyze-btn:hover,
button.report-btn:hover {
    background-color: #0D47A1;
}
"""

# === Gradio UI ===
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>🧪 Cervical Cytology AI Analysis Tool</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload Slide and Enter Patient Info")

            with gr.Row():
                patient_id = gr.Textbox(label="Patient ID")
                slide_id = gr.Textbox(label="Slide ID")
            image_input = gr.Image(label="Upload Slide Image (LBC)", type="pil")

            analyze_btn = gr.Button("🔍 Analyze Slide", elem_classes="analyze-btn")
            alert_output = gr.Textbox(label="Alert Message", lines=1, interactive=False)
            llm_output = gr.Textbox(label="LLM-generated Analysis", lines=6, interactive=False)

        with gr.Column(scale=1):
            with gr.Row():
                patient_id_display = gr.Textbox(label="Patient ID", interactive=False)
                slide_id_display = gr.Textbox(label="Slide ID", interactive=False)
            with gr.Row():
                date_output = gr.Textbox(label="Date", interactive=False)
                prediction_output = gr.Textbox(label="Prediction", interactive=False)

            image_output = gr.Image(label="Overlayed Cytology Image")
            cell_dist_output = gr.Textbox(label="Cell Type Distribution", lines=6, interactive=False)

            remarks = gr.Textbox(label="Remarks")
            report_btn = gr.Button("📄 Generate Report", elem_classes="report-btn")

    original_image_state = gr.State()
    overlay_image_state = gr.State()

    analyze_btn.click(
        fn=analyze_slide,
        inputs=[patient_id, slide_id, image_input],
        outputs=[
            date_output,
            patient_id_display,
            slide_id_display,
            prediction_output,
            image_output,
            alert_output,
            cell_dist_output,
            llm_output,
            original_image_state,
            overlay_image_state
        ]
    )

    report_btn.click(
        fn=generate_html_report,
        inputs=[
            date_output,
            patient_id_display,
            slide_id_display,
            prediction_output,
            alert_output,
            cell_dist_output,
            llm_output,
            original_image_state,
            overlay_image_state,
            remarks
        ],
        outputs=[]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
