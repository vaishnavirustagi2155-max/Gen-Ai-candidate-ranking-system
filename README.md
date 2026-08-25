\# KRAM AI — GenAI Candidate Ranking System



> An AI-powered candidate screening and ranking platform that analyzes job descriptions and resumes to identify the best-fit candidates.



\## 🚀 Live Demo



\*\*Website:\*\* https://kram-omega.vercel.app/



\---



\## 📌 Overview



KRAM AI is a GenAI-powered candidate ranking system designed to make resume screening faster and more data-driven.



The platform allows recruiters to:



\- Enter a job description

\- Upload multiple candidate resumes

\- Extract candidate information and skills

\- Compare candidates against job requirements

\- Calculate candidate match scores

\- Rank candidates automatically

\- View matched and missing skills

\- Analyze semantic similarity between resumes and job descriptions

\- View explainable scoring factors using SHAP

\- Explore candidate analytics

\- Generate candidate reports



The system combines traditional skill matching with semantic analysis and explainable AI to provide a more comprehensive candidate ranking.



\---



\## ✨ Key Features



\### 📄 Resume Processing



\- Upload multiple resumes

\- PDF resume parsing

\- Candidate name and contact extraction

\- Skill extraction

\- Experience extraction

\- Resume text processing



\### 🎯 Intelligent Candidate Matching



Candidates are evaluated using multiple factors:



\- Required skill matching

\- Preferred skill matching

\- Experience matching

\- Semantic similarity

\- Overall candidate score



\### 🧠 Semantic Matching



The system uses semantic analysis to compare the meaning of:



\*\*Job Description ↔ Candidate Resume\*\*



This allows candidates to receive relevant matches even when their resume wording does not exactly match the job description.



\### 🔍 Explainable AI



KRAM AI provides SHAP-based explanations showing which factors influenced a candidate's ranking.



Examples include:



\- Required Skills

\- Preferred Skills

\- Experience

\- Semantic Similarity



This helps recruiters understand \*\*why\*\* a candidate received a particular score.



\### 📊 Analytics Dashboard



The platform provides analytics such as:



\- Candidate score distribution

\- Average candidate score

\- Top candidates

\- Skill coverage

\- Match statistics

\- Candidate ranking insights



\### 📑 Candidate Reports



Candidate information and ranking results can be viewed through dedicated candidate and report pages.



\---



\## 🏗️ System Architecture



```text

&#x20;                   ┌──────────────────────┐

&#x20;                   │      KRAM AI         │

&#x20;                   │      Frontend        │

&#x20;                   │     Next.js          │

&#x20;                   └──────────┬───────────┘

&#x20;                              │

&#x20;                              │ REST API

&#x20;                              ▼

&#x20;                   ┌──────────────────────┐

&#x20;                   │       FastAPI        │

&#x20;                   │       Backend        │

&#x20;                   └──────────┬───────────┘

&#x20;                              │

&#x20;            ┌─────────────────┼─────────────────┐

&#x20;            │                 │                 │

&#x20;            ▼                 ▼                 ▼

&#x20;      Resume Parser      Skill Matching    Semantic Matching

&#x20;            │                 │                 │

&#x20;            └─────────────────┼─────────────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌──────────────────────┐

&#x20;                   │ Candidate Ranking    │

&#x20;                   │ \& SHAP Explanation   │

&#x20;                   └──────────────────────┘

