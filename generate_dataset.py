import csv
import random

random.seed(42)

CATEGORIES = {
    "Data Scientist": {
        "skills": ["Python", "pandas", "NumPy", "scikit-learn", "TensorFlow", "SQL",
                   "machine learning", "data visualization", "statistics", "A/B testing",
                   "deep learning", "feature engineering", "Tableau", "Power BI", "NLP"],
        "roles": ["Data Scientist", "ML Intern", "Data Analyst"],
        "verbs": ["built predictive models", "analyzed large datasets", "trained classification models",
                  "designed dashboards", "ran statistical experiments", "cleaned and preprocessed data"],
    },
    "Web Developer": {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "REST APIs",
                   "MongoDB", "Express.js", "Git", "responsive design", "Redux", "TypeScript"],
        "roles": ["Frontend Developer", "Full Stack Developer", "Web Developer Intern"],
        "verbs": ["built responsive websites", "developed REST APIs", "optimized page load speed",
                  "implemented UI components", "integrated third-party APIs", "fixed cross-browser bugs"],
    },
    "Software Engineer": {
        "skills": ["Java", "C++", "Python", "Data Structures", "Algorithms", "Git",
                   "system design", "OOP", "multithreading", "unit testing", "Docker", "Kubernetes"],
        "roles": ["Software Engineer", "SDE Intern", "Backend Engineer"],
        "verbs": ["designed scalable systems", "wrote unit tests", "debugged production issues",
                  "reviewed pull requests", "optimized algorithms", "built microservices"],
    },
    "HR": {
        "skills": ["recruitment", "onboarding", "employee engagement", "payroll",
                   "HRIS", "conflict resolution", "performance management", "labor law",
                   "talent acquisition", "training and development"],
        "roles": ["HR Executive", "Talent Acquisition Specialist", "HR Generalist"],
        "verbs": ["managed end-to-end recruitment", "conducted employee onboarding",
                  "handled payroll processing", "organized training programs", "resolved workplace conflicts"],
    },
    "Business Analyst": {
        "skills": ["Excel", "SQL", "requirements gathering", "stakeholder management",
                   "process improvement", "Power BI", "JIRA", "business process modeling", "SWOT analysis"],
        "roles": ["Business Analyst", "Business Analyst Intern", "Product Analyst"],
        "verbs": ["gathered business requirements", "created process flow diagrams",
                  "presented insights to stakeholders", "analyzed KPIs", "coordinated with cross-functional teams"],
    },
    "Mechanical Engineer": {
        "skills": ["AutoCAD", "SolidWorks", "thermodynamics", "CATIA", "manufacturing processes",
                   "CNC machining", "GD&T", "finite element analysis", "quality control"],
        "roles": ["Mechanical Design Engineer", "Manufacturing Engineer", "Mechanical Intern"],
        "verbs": ["designed mechanical components", "performed stress analysis",
                  "supervised production lines", "created CAD models", "reduced material waste"],
    },
    "Sales": {
        "skills": ["CRM", "lead generation", "negotiation", "cold calling", "Salesforce",
                   "client relationship management", "target achievement", "market research"],
        "roles": ["Sales Executive", "Business Development Associate", "Sales Intern"],
        "verbs": ["exceeded quarterly sales targets", "built client relationships",
                  "generated new leads", "negotiated contracts", "managed key accounts"],
    },
    "Marketing": {
        "skills": ["SEO", "content marketing", "Google Analytics", "social media marketing",
                   "email campaigns", "brand strategy", "Google Ads", "market research", "copywriting"],
        "roles": ["Digital Marketing Executive", "Marketing Intern", "Social Media Manager"],
        "verbs": ["ran paid ad campaigns", "grew social media following", "increased organic traffic",
                  "wrote marketing copy", "analyzed campaign performance"],
    },
    "Java Developer": {
        "skills": ["Java", "Spring Boot", "Hibernate", "REST APIs", "Microservices",
                   "MySQL", "Maven", "JUnit", "Kafka", "multithreading"],
        "roles": ["Java Developer", "Backend Developer", "Java Intern"],
        "verbs": ["built REST APIs using Spring Boot", "optimized database queries",
                  "implemented microservices", "wrote JUnit test cases", "resolved production bugs"],
    },
    "DevOps Engineer": {
        "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Jenkins", "Terraform",
                   "Linux", "monitoring", "Ansible", "shell scripting"],
        "roles": ["DevOps Engineer", "Cloud Engineer", "DevOps Intern"],
        "verbs": ["built CI/CD pipelines", "automated infrastructure with Terraform",
                  "managed Kubernetes clusters", "set up monitoring and alerting", "reduced deployment time"],
    },
}

EDUCATION = [
    "B.Tech in Computer Engineering", "B.E. in Information Technology",
    "B.Sc in Computer Science", "MBA", "B.Tech in Mechanical Engineering",
    "Diploma in Engineering", "M.Tech in Data Science",
]


def make_resume(category, info):
    """Create one synthetic resume text for a given category."""
    skills = random.sample(info["skills"], k=min(6, len(info["skills"])))
    verbs = random.sample(info["verbs"], k=min(3, len(info["verbs"])))
    role = random.choice(info["roles"])
    edu = random.choice(EDUCATION)
    years = random.choice(["0", "1", "2", "3", "4", "5"])

    return (
        f"{role} with {years} years of experience. "
        f"Education: {edu}. "
        f"Skills: {', '.join(skills)}. "
        f"Experience: {'; '.join(verbs)}. "
        f"Looking for opportunities as a {role.lower()} to apply skills in "
        f"{', '.join(skills[:3])}."
    )


def main():
    rows = []

    # Create 40 resumes per category = 400 total resumes
    for category, info in CATEGORIES.items():
        for _ in range(40):
            rows.append({
                "resume_text": make_resume(category, info),
                "category": category,
            })

    random.shuffle(rows)

    with open("data/resume_dataset.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["resume_text", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows across {len(CATEGORIES)} categories to data/resume_dataset.csv")


if __name__ == "__main__":
    main()
