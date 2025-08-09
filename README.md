
# ADGM Corporate Agent — Upgraded Prototype

This project is an **offline document review assistant** that processes `.docx` legal documents for **ADGM Corporate Agent** workflows.  
It performs automated compliance checks, detects missing documents, flags red-flag clauses, and attaches relevant ADGM reference citations.  

It meets all the requirements of the given **AI Task** by:
- Detecting document types and mapping them to ADGM checklists
- Identifying missing required documents
- Performing rule-based red-flag detection
- Suggesting corrective actions
- Performing **offline Retrieval-Augmented Generation (RAG)** using provided reference PDFs
- Producing:
  1. **Reviewed `.docx`** — Original document with appended review notes and ADGM citations  
  2. **JSON report** — Structured details of findings

## 📂 Project Structure
ADGM_CorporateAgent_Main.py # Main Python application
Task.pdf # Reference document
Data Sources.pdf # Reference document
requirements.txt # Python dependencies
README.md # This file

## ⚙️ Installation

1. **Clone or download** this project to your local machine.
2. Open a terminal/command prompt in the project folder.
3. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
4.  Install dependencies:
```bash
pip install -r requirements.txt
```
**▶️ Running the Application**
Run the following command:
```bash
python ADGM_CorporateAgent_Main.py
```
This will start a Gradio web application.
Open the local URL shown in the terminal (e.g., http://127.0.0.1:7860) in your browser.

📋 Usage Instructions
1. In the web interface, upload a .docx legal document.
2. Click Run Review.
3. The system will:
   -Identify the legal process and required documents
   -Detect missing documents
   -Flag red-flag issues with severity levels and suggestions
   -Search ADGM reference PDFs for relevant guidance
4. Download the reviewed .docx containing notes and citations.
5. View or copy the JSON structured report.

.

**📑 Example JSON Output**

{
  "process": "Company Incorporation",
  "documents_detected": [
    "Articles of Association",
    "Incorporation Application Form"
  ],
  "documents_uploaded": 2,
  "required_documents": 5,
  "missing_documents": [
    "Memorandum of Association",
    "UBO Declaration Form",
    "Register of Members and Directors"
  ],
  "issues_found": [
    {
      "section": "UBO",
      "issue": "No UBO (Ultimate Beneficial Owner) declaration found",
      "severity": "High",
      "suggestion": "Include a UBO declaration in accordance with ADGM requirements."
    }
  ],
  "rag_matches": {
    "0": [
      {
        "score": 0.456,
        "source": "Task.pdf",
        "page": 5,
        "excerpt": "ADGM requires all entities to submit a UBO declaration..."
      }
    ]
  }
}

**📷 Example Output Screenshot**

The screenshot below shows the ADGM Corporate Agent — Upgraded Prototype interface after running a compliance check on a sample .docx legal document.
**Key Parts of the Screenshot**
1. Upload Section
   - A .docx legal document has been uploaded:
    adgm-ra-resolution-multiple-incorporate-shareholders-LTD-incorporation-v2 (1).docx
2. Reviewed Document Download
   - The reviewed .docx file is available for download:
     reviewed_1754694106_adgm-ra-resolution-multiple-incorporate-shareholders-LTD-incorporation
     v2 1.docx
   - This file contains inline review notes and ADGM citations.
3. Structured Report (JSON Output)
   - Process Detected:
      "Company Incorporation" — The system identified the process type based on document content.
   - Documents Detected:
    "Articles of Association", "Incorporation Application Form"
   - Documents Uploaded Count:
     2 out of 5 required documents were detected.
   - Missing Documents:
     "Memorandum of Association", "UBO Declaration Form", "Register of Members and Directors"
   - Issues Found:
         Section: "UBO"
         Issue: "No UBO (Ultimate Beneficial Owner) declaration found"
         Severity: "High"
         Suggestion: "Include a UBO declaration in accordance with ADGM requirements."
This example demonstrates how the system:
   - Detects the type of process
   - Checks for completeness of the required document set
   - Flags compliance gaps
   - Suggests corrective actions with ADGM-specific references
<img width="1884" height="924" alt="image" src="https://github.com/user-attachments/assets/b6e424f4-e0bc-4876-8140-36280417320d" />

