# KRAM AI – GenAI Candidate Ranking System

KRAM AI is a candidate screening and ranking web application that helps recruiters compare resumes against a given job description.

The idea behind this project was to make the initial resume screening process faster by automatically extracting information from resumes, matching skills with the job requirements, and ranking candidates based on how well they fit the role.

## Live Demo

🌐 **Website:** https://kram-omega.vercel.app/

---

## What does KRAM AI do?

A recruiter can enter a job description and upload multiple candidate resumes.

The system then:

- Extracts candidate information from resumes
- Identifies skills and experience
- Extracts required and preferred skills from the job description
- Compares candidates with the job requirements
- Calculates a score for each candidate
- Ranks candidates from best to lowest match
- Shows matched and missing skills
- Calculates semantic similarity between the resume and job description
- Provides an explanation of the factors affecting the score
- Shows analytics for the uploaded candidates

The goal is not to completely replace human recruiters, but to make the first stage of candidate screening easier and faster.

---

## How the ranking works

Each candidate gets an overall score based on four main factors:

- **Required Skills – 40%**
- **Preferred Skills – 15%**
- **Experience – 20%**
- **Semantic Similarity – 25%**

For example, if a candidate has most of the required skills but doesn't have enough experience, the system reflects both things in the final score.

The semantic matching part also helps when a resume uses different wording from the job description but is still relevant to the role.

---

## Explainable AI

One of the parts I wanted to include in this project was an explanation of **why** a candidate received a particular score.

KRAM AI uses SHAP-based explanations to show the factors that affected the candidate's ranking.

For example:

```text
Required Skills       +14.28
Experience             +10.00
Semantic Similarity    -1.47
Preferred Skills       -7.50
