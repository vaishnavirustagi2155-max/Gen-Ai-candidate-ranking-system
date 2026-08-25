# KRAM AI – GenAI Candidate Ranking System

KRAM AI is a web-based candidate screening and ranking system that helps recruiters compare resumes with a given job description.

I built this project to explore how AI, NLP, semantic matching, and explainable AI can be combined to make the initial resume screening process faster and more structured.

Instead of manually going through every resume, a recruiter can enter a job description, upload multiple resumes, and get a ranked list of candidates based on their overall match.

## Live Demo

🌐 **KRAM AI:** https://kram-omega.vercel.app/

💻 **GitHub Repository:** https://github.com/vaishnavirustagi2155-max/Gen-Ai-candidate-ranking-system

---

## What KRAM AI does

The application takes a job description and a set of candidate resumes and processes them automatically.

The main workflow is:

1. Enter a job description.
2. Upload candidate resumes.
3. Extract candidate information from the resumes.
4. Extract required and preferred skills from the job description.
5. Compare candidate skills with the job requirements.
6. Evaluate candidate experience.
7. Calculate semantic similarity between the job description and resume.
8. Generate an overall candidate score.
9. Rank all candidates.
10. Show why a candidate received their score.
11. Display analytics and candidate-level information.

The idea is to help recruiters quickly identify the most relevant candidates while still keeping the final hiring decision with the recruiter.

---

## Main Features

### Resume Upload

Recruiters can upload multiple PDF resumes for a single job opening.

The system processes each resume and extracts useful information such as:

- Candidate name
- Email
- Phone number
- Skills
- Experience
- Resume text

---

### Job Description Analysis

The system analyzes the provided job description and identifies:

- Required skills
- Preferred skills
- Minimum experience
- Job description content

For example, a Data Analyst job may require:

```text
Python
SQL
Power BI
Excel
Pandas
NumPy
Data Analysis
