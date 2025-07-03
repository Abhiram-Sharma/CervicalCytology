import base64
import requests
import json
import os
 
# === CONFIGURATION ===
GEMINI_API_KEY = "AIzaSyC7U-TYlgsUT40nZCWZSglpWSjeRis_Ecw"  # 🔐 Replace this with your actual API key
GEMINI_MODEL = "gemini-1.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
 
# === FUNCTION TO ENCODE IMAGE TO BASE64 ===
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
 
# === FUNCTION TO SEND REQUEST TO GEMINI ===
def query_gemini_with_image_and_prompt(image_path, prompt):
    b64_image = encode_image_to_base64(image_path)
 
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image
                        }
                    }
                ]
            }
        ]
    }
 
    headers = {"Content-Type": "application/json"}
 
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "[⚠️] Gemini returned an unexpected response format."
    else:
        return f"[❌] Error {response.status_code}: {response.text}"
def LLM_Analysis(predicted,image):
    image_file=image
    answer=predicted
    prompt_text = "Please review this overlayed image like a medical professional, the original whole‐field LBC slide showing a cluster of stained cervical squamous cells. Based on the model’s prediction of "+answer+", generate a concise, professional medical diagnosis report in one paragraph with both a quantitavie and qualitative analysis having atleast 3 significant numbers that: (1) describes any visible cytoplasmic features in the segmented cell, (2) correlates these with nuclear and architectural findings observed in the full‐field image (e.g., hyperchromasia, irregular nuclear contours, pleomorphism, loss of polarity), (3) states the classification as "+answer+", and (4) concludes with recommended clinical follow‐up (colposcopy, biopsy) to confirm moderate‑to‑severe conditions and prevent progression to invasive carcinoma."
    result = query_gemini_with_image_and_prompt(image_file, prompt_text)
    return result
 
# === MAIN EXECUTION ===
if __name__ == "__main__":
    # 👇 Change this to your image file path
    image_file = "final_overlay_output.png"  # Make sure the file exists in your current directory
 
    # 🧠 Example prompt
    answer="High grade"
    prompt_text = "Please review this overlayed image like a medical professional, the original whole‐field LBC slide showing a cluster of stained cervical squamous cells. Based on the model’s prediction of "+answer+", generate a concise, professional medical diagnosis report in one paragraph with both a quantitavie and qualitative analysis having atleast 3 significant numbers that: (1) describes any visible cytoplasmic features in the segmented cell, (2) correlates these with nuclear and architectural findings observed in the full‐field image (e.g., hyperchromasia, irregular nuclear contours, pleomorphism, loss of polarity), (3) states the classification as "+answer+", and (4) concludes with recommended clinical follow‐up (colposcopy, biopsy) to confirm moderate‑to‑severe conditions and prevent progression to invasive carcinoma."
 
    print("Analysing...")
    result = query_gemini_with_image_and_prompt(image_file, prompt_text)
    print("\nReport:\n")
    print(result)