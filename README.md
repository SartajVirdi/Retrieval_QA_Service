#  LLM-Powered Insurance Document Query System

This project is built for **HackRx 6.0** and allows users to ask natural language questions (like insurance claims) and receive structured decisions extracted from complex documents such as health policies.

---

##  API Endpoint

**POST** `/api/v1/hackrx/run`

###  Request Format

```json
{
  "query": "46M, knee surgery, Pune, 3-month policy",
  "chunks": [
    "Bajaj Allianz provides coverage for knee surgery after 90 days of policy activation.",
    "Applicable to males aged 18–60 across major cities including Pune.",
    "Day-care procedures and hospitalizations are included under the Gold Health Plan."
  ]
}
