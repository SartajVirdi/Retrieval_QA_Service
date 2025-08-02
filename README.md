#  HackRx 6.0 – Intelligent Policy Query API

This project implements a lightweight and accurate PDF-based Q&A API for insurance policy documents, designed to meet the **HackRx 6.0 Problem Statement Q1**:

> "Design an LLM-powered Intelligent Query–Retrieval System that processes large documents (PDF, DOCX, emails), matches relevant clauses, and returns explainable answers."

---

##  Features

- ✅ Accepts **PDF URLs** of insurance policy documents
- ✅ Accepts **multiple natural-language questions**
- ✅ Uses **Google Gemini 2.0 Flash** for fast and accurate responses
- ✅ FastAPI-based `/hackrx/run` endpoint
- ✅ Deployment-ready for Render or any cloud platform

---

##  Installation

```bash
git clone https://github.com/<your-team>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```
---
###  Environment Variables
- Create a .env file or set your environment variables as:
- GEMINI_API_KEY=your_gemini_api_key_here
- To generate a Gemini API key: https://makersuite.google.com/app/apikey
---

###  API Endpoint
POST /hackrx/run

Request Body:
```bash
{
  "documents": "<PDF_URL>",
  "questions": [
    "What is the grace period?",
    "Does the policy cover maternity expenses?"
  ]
}
```
Response Body:
```bash
{
  "answers": [
    "The grace period is 30 days from due date...",
    "Yes, maternity expenses are covered after 9 months..."
  ]
}
```
---
###  Example Test (using cURL) 
```bash
curl -X POST https://your-app.onrender.com/hackrx/run \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the waiting period for pre-existing conditions?",
      "Does this policy offer cashless hospitalization?"
    ]
}'
```
---
 Submission Criteria Met
 - /hackrx/run POST endpoint

 - Accepts a PDF URL and a list of questions

 - Returns answers in clean structured JSON

 - Based on real LLM (Gemini 2.0 Flash)

 - Lightweight, accurate, explainable, deployable
 
---
###  Authors
- Team Name: AVENGERS_404

- Sartaj Singh Virdi

- Gurkirat Singh

- Prabhpreet Singh

###  License

MIT License
