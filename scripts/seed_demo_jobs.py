"""One-off seeder for demo companies + published jobs, for founder review.

Bypasses the chat/LLM pipeline entirely (writes job rows + JD content directly via the same
database functions the graph nodes use) — this is bulk demo data, not a real recruiter
conversation, so there's no reason to burn Mistral calls (and risk 429s) generating it turn by
turn. Idempotent: re-running (or importing seed() from main.py's startup hook) silently skips
any company that already exists rather than duplicating data.

Usage:
    python scripts/seed_demo_jobs.py

Also importable as `from scripts.seed_demo_jobs import seed` — see BOOTSTRAP_DEMO_JOBS in
backend/config.py for running this automatically on app startup (Railway has no working SSH
path, same reason BOOTSTRAP_ADMIN_* exists).
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.auth import hash_password
from backend.database import (
    create_company_and_recruiter,
    finalize_publish,
    init_db,
    save_jd_versions,
    save_selected_version,
    upsert_job_draft,
)

COMPANIES = [
    {
        "company_name": "Amazon",
        "recruiter_email": "default@gmail.com",
        "recruiter_password": "default1234",
        "recruiter_name": "Alex Morgan",
        "profile": {
            "industry": "E-commerce & Cloud Computing",
            "company_overview": (
                "Amazon is a global technology company driving innovation in e-commerce, cloud "
                "computing, digital streaming, and artificial intelligence. We operate at a scale "
                "unmatched in the industry, serving hundreds of millions of customers and "
                "businesses worldwide."
            ),
            "website": "https://www.amazon.jobs",
            "headquarters": "Seattle, WA, USA",
            "company_culture": (
                "We operate on a set of Leadership Principles that shape every decision, from the "
                "smallest to the biggest. We're a company of builders who are always inventing, "
                "iterating, and improving on behalf of customers."
            ),
            "benefits": (
                "Comprehensive health coverage, 401(k) matching, stock unit awards (RSUs), "
                "parental leave, and career development programs including Amazon Career Choice."
            ),
            "work_life_balance": (
                "We support flexible working arrangements including hybrid and remote options for "
                "many roles, along with generous PTO and wellness resources."
            ),
            "why_join_us": (
                "Join a team that's redefining what's possible at scale — work on problems that "
                "affect millions of customers daily, with the resources and autonomy to innovate "
                "quickly."
            ),
        },
    },
    {
        "company_name": "Google",
        "recruiter_email": "recruiter@googledemo.example",
        "recruiter_password": "GoogleDemo1234",
        "recruiter_name": "Priya Anand",
        "profile": {
            "industry": "Technology & Internet Services",
            "company_overview": (
                "Google builds products and platforms used by billions of people every day, from "
                "Search and Android to Cloud and Workspace. We invest heavily in research, "
                "infrastructure, and talent to solve problems at global scale."
            ),
            "website": "https://careers.google.com",
            "headquarters": "Mountain View, CA, USA",
            "company_culture": (
                "A culture of intellectual curiosity, data-driven decision making, and "
                "collaborative problem solving — we value fresh perspectives at every level."
            ),
            "benefits": (
                "Comprehensive medical/dental/vision, generous parental leave, on-site wellness "
                "and fitness facilities, equity compensation, and continuous learning stipends."
            ),
            "work_life_balance": "Flexible hybrid work models, unlimited sick time, and a strong emphasis on sustainable pace.",
            "why_join_us": (
                "Work alongside some of the best engineers and researchers in the world on "
                "products that shape how the world accesses information."
            ),
        },
    },
    {
        "company_name": "Netflix",
        "recruiter_email": "recruiter@netflixdemo.example",
        "recruiter_password": "NetflixDemo1234",
        "recruiter_name": "Jordan Lee",
        "profile": {
            "industry": "Entertainment & Streaming",
            "company_overview": (
                "Netflix is the world's leading streaming entertainment service, with hundreds of "
                "millions of members in over 190 countries. We produce and license a wide variety "
                "of TV series, films, and games."
            ),
            "website": "https://jobs.netflix.com",
            "headquarters": "Los Gatos, CA, USA",
            "company_culture": (
                "We operate with high performance and high trust — freedom and responsibility are "
                "core to how we work, with minimal process and maximum context-sharing."
            ),
            "benefits": "Unlimited vacation policy, top-of-market compensation, comprehensive health coverage, and generous parental leave.",
            "work_life_balance": "We trust employees to manage their own schedules and focus on outcomes over hours logged.",
            "why_join_us": (
                "Join a company that trusts you to do the best work of your career, with the "
                "context and autonomy to make high-impact decisions."
            ),
        },
    },
]

AMAZON_JOBS = [
    dict(
        job_title="Software Development Engineer II",
        job_category="Software Engineering",
        experience="2-3 years",
        location="Bangalore",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="₹28,00,000 - ₹42,00,000 / year",
        required_skills=["Java", "Distributed Systems", "AWS", "Data Structures & Algorithms"],
        preferred_skills=["Kotlin", "DynamoDB", "Kubernetes"],
        responsibilities=[
            "Design and build scalable backend services powering Amazon's retail platform",
            "Own features end-to-end, from design review through production monitoring",
            "Participate in on-call rotation and drive operational excellence",
        ],
        job_summary=(
            "We're looking for a Software Development Engineer II to design and build highly "
            "scalable, distributed backend services that power core parts of Amazon's retail "
            "platform, serving millions of customers every day."
        ),
        about_role=(
            "As an SDE II, you'll work within a small, autonomous team that owns its services "
            "end-to-end — from design through production operations. You'll partner closely with "
            "product managers and senior engineers to translate ambiguous customer problems into "
            "well-architected, reliable systems, while mentoring junior engineers along the way."
        ),
        major_accountabilities=[
            "Design, implement, and operate distributed services at Amazon scale",
            "Drive technical design reviews and contribute to architecture decisions",
            "Write high-quality, well-tested code and review peers' contributions",
            "Participate in an on-call rotation and respond to operational issues",
            "Mentor junior engineers and contribute to a strong engineering culture",
        ],
        minimum_requirements=[
            "2-3 years of professional software development experience",
            "Strong proficiency in Java or an equivalent object-oriented language",
            "Solid understanding of data structures, algorithms, and distributed systems fundamentals",
            "Experience building and operating production services",
        ],
        required_qualifications=[
            "Bachelor's degree in Computer Science or a related field",
            "Experience with AWS or another major cloud platform",
        ],
        preferred_qualifications=[
            "Experience with Kotlin",
            "Familiarity with DynamoDB or other NoSQL data stores",
            "Exposure to container orchestration (Kubernetes/ECS)",
        ],
        stand_out=[
            "Contributions to open-source projects",
            "Experience operating services at high scale (millions of requests/day)",
        ],
        benefits=["Stock unit awards (RSUs)", "401(k) matching", "Comprehensive health coverage", "Career development programs"],
    ),
    dict(
        job_title="Senior Data Scientist",
        job_category="Data / Analytics",
        experience="4-6 years",
        location="Seattle, WA",
        work_mode="Onsite",
        employment_type="Full-time",
        education="Master's degree",
        salary="$150,000 - $190,000 / year",
        required_skills=["Python", "SQL", "Machine Learning", "Statistical Modeling"],
        preferred_skills=["PySpark", "A/B Testing", "Deep Learning"],
        responsibilities=[
            "Build predictive models to improve demand forecasting and inventory planning",
            "Partner with product and engineering teams to deploy models into production",
            "Design and analyze large-scale experiments (A/B tests)",
        ],
        job_summary=(
            "Amazon is looking for a Senior Data Scientist to build predictive models that improve "
            "demand forecasting, inventory placement, and customer experience across our retail "
            "operations."
        ),
        about_role=(
            "You'll work with petabyte-scale datasets to develop, validate, and deploy machine "
            "learning models that directly influence business decisions. This role partners "
            "closely with engineering and product teams to move models from prototype to "
            "production, and requires strong statistical rigor alongside pragmatic engineering "
            "judgment."
        ),
        major_accountabilities=[
            "Design, build, and validate machine learning models for forecasting and optimization",
            "Translate ambiguous business problems into well-scoped data science projects",
            "Partner with engineering to productionize models at scale",
            "Design and analyze A/B tests to measure business impact",
            "Communicate findings and recommendations to senior stakeholders",
        ],
        minimum_requirements=[
            "4-6 years of experience in applied data science or machine learning",
            "Strong Python and SQL skills",
            "Solid grounding in statistical modeling and experimental design",
        ],
        required_qualifications=[
            "Master's degree in a quantitative field (Statistics, CS, Operations Research, or related)",
            "Track record of shipping ML models into production systems",
        ],
        preferred_qualifications=[
            "Experience with PySpark or other large-scale data processing frameworks",
            "Familiarity with deep learning frameworks (PyTorch/TensorFlow)",
        ],
        stand_out=["Published research or patents in ML/statistics", "Experience with supply chain or forecasting domains"],
        benefits=["Stock unit awards (RSUs)", "401(k) matching", "Comprehensive health coverage", "Relocation assistance"],
    ),
    dict(
        job_title="Program Manager, Operations",
        job_category="Operations",
        experience="4-6 years",
        location="Hyderabad",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="₹22,00,000 - ₹32,00,000 / year",
        required_skills=["Program Management", "Process Improvement", "Data Analysis", "Stakeholder Management"],
        preferred_skills=["Six Sigma", "SQL", "Supply Chain Operations"],
        responsibilities=[
            "Drive cross-functional programs that improve fulfillment center efficiency",
            "Define metrics, track performance, and report to senior leadership",
            "Identify and eliminate operational bottlenecks",
        ],
        job_summary=(
            "We're hiring a Program Manager to drive operational excellence initiatives across "
            "Amazon's fulfillment network, working cross-functionally to improve efficiency, "
            "quality, and customer experience."
        ),
        about_role=(
            "This role sits at the intersection of operations, engineering, and business teams. "
            "You'll own programs from problem definition through measurable results, using data "
            "to prioritize investments and drive continuous improvement across a large, complex "
            "operational network."
        ),
        major_accountabilities=[
            "Own end-to-end delivery of cross-functional operational improvement programs",
            "Define success metrics and build reporting to track progress",
            "Partner with engineering, finance, and operations stakeholders",
            "Identify root causes of inefficiency and drive corrective action plans",
        ],
        minimum_requirements=[
            "4-6 years of program or project management experience",
            "Strong analytical skills with comfort working in data",
            "Excellent written and verbal communication skills",
        ],
        required_qualifications=["Bachelor's degree", "Experience managing multiple concurrent workstreams"],
        preferred_qualifications=["Six Sigma or Lean certification", "Working SQL proficiency", "Supply chain or logistics background"],
        stand_out=["Experience in a high-volume fulfillment or manufacturing environment"],
        benefits=["Stock unit awards (RSUs)", "Comprehensive health coverage", "Career development programs"],
    ),
    dict(
        job_title="Solutions Architect, AWS",
        job_category="Cloud / Infrastructure",
        experience="7+ years",
        location="Worldwide",
        work_mode="Remote",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="$160,000 - $210,000 / year",
        required_skills=["AWS", "Cloud Architecture", "Networking", "Security"],
        preferred_skills=["Terraform", "Kubernetes", "Enterprise Sales Support"],
        responsibilities=[
            "Design and recommend AWS cloud architectures for enterprise customers",
            "Serve as a trusted technical advisor throughout the customer's cloud journey",
            "Deliver technical workshops and architecture reviews",
        ],
        job_summary=(
            "Amazon Web Services is looking for a Solutions Architect to help enterprise customers "
            "design and implement secure, scalable, cost-effective cloud architectures on AWS."
        ),
        about_role=(
            "You'll be the primary technical point of contact for a portfolio of enterprise "
            "accounts, guiding customers through architecture design, migration planning, and "
            "ongoing optimization. This role blends deep technical expertise with strong "
            "communication skills to translate business needs into sound technical solutions."
        ),
        major_accountabilities=[
            "Design well-architected, secure, and cost-optimized AWS solutions for customers",
            "Lead technical workshops, architecture reviews, and proof-of-concepts",
            "Build long-term trusted relationships with customer engineering and leadership teams",
            "Stay current with new AWS services and industry best practices",
        ],
        minimum_requirements=[
            "7+ years of experience in cloud architecture or infrastructure engineering",
            "Deep hands-on experience with AWS services (compute, storage, networking, security)",
            "Strong customer-facing communication skills",
        ],
        required_qualifications=["Bachelor's degree in Computer Science, Engineering, or equivalent experience"],
        preferred_qualifications=["AWS certifications (Solutions Architect Professional preferred)", "Experience with Terraform or CloudFormation"],
        stand_out=["Prior experience in a customer-facing technical role (pre-sales, consulting)"],
        benefits=["Stock unit awards (RSUs)", "Fully remote flexibility", "Comprehensive health coverage"],
    ),
    dict(
        job_title="Business Analyst, Retail Analytics",
        job_category="Data / Analytics",
        experience="2-3 years",
        location="Bangalore",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="₹14,00,000 - ₹20,00,000 / year",
        required_skills=["SQL", "Excel", "Data Visualization", "Business Analysis"],
        preferred_skills=["Tableau", "Python", "A/B Testing"],
        responsibilities=[
            "Analyze retail sales and customer behavior data to surface actionable insights",
            "Build dashboards and reports for category and merchandising teams",
            "Partner with stakeholders to define and track key business metrics",
        ],
        job_summary=(
            "We're looking for a Business Analyst to turn large volumes of retail data into "
            "clear, actionable insights that guide merchandising and category management "
            "decisions."
        ),
        about_role=(
            "You'll work closely with category managers and business leaders to understand their "
            "questions, build the analysis and dashboards to answer them, and present "
            "recommendations backed by data. This is a great role for someone who enjoys both the "
            "technical and communication sides of analytics."
        ),
        major_accountabilities=[
            "Write SQL queries to extract and analyze large retail datasets",
            "Build and maintain dashboards tracking key business metrics",
            "Present findings and recommendations to business stakeholders",
            "Support experiment design and results analysis",
        ],
        minimum_requirements=["2-3 years of experience in business or data analysis", "Strong SQL skills", "Comfort working with large datasets"],
        required_qualifications=["Bachelor's degree in a quantitative or business field"],
        preferred_qualifications=["Experience with Tableau or similar BI tools", "Working knowledge of Python for analysis"],
        stand_out=["Experience in e-commerce or retail analytics specifically"],
        benefits=["Stock unit awards (RSUs)", "Comprehensive health coverage", "Career development programs"],
    ),
    dict(
        job_title="DevOps Engineer",
        job_category="Cloud / Infrastructure",
        experience="2-3 years",
        location="Dublin",
        work_mode="Onsite",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="€65,000 - €85,000 / year",
        required_skills=["AWS", "CI/CD", "Terraform", "Linux"],
        preferred_skills=["Kubernetes", "Python", "Monitoring & Observability"],
        responsibilities=[
            "Build and maintain CI/CD pipelines for service deployment",
            "Manage infrastructure as code across multiple AWS accounts",
            "Improve system reliability, monitoring, and incident response",
        ],
        job_summary=(
            "We're hiring a DevOps Engineer to build and maintain the infrastructure and deployment "
            "pipelines that keep Amazon's services running reliably at scale."
        ),
        about_role=(
            "You'll own infrastructure-as-code, CI/CD tooling, and observability for a portfolio of "
            "services, working closely with development teams to make deployments fast, safe, and "
            "repeatable. This role has strong ownership of production reliability."
        ),
        major_accountabilities=[
            "Design and maintain CI/CD pipelines for automated build, test, and deployment",
            "Manage infrastructure as code using Terraform or CloudFormation",
            "Improve monitoring, alerting, and incident response processes",
            "Participate in an on-call rotation supporting production systems",
        ],
        minimum_requirements=["2-3 years of DevOps or infrastructure engineering experience", "Hands-on experience with AWS", "Strong Linux fundamentals"],
        required_qualifications=["Bachelor's degree in Computer Science or related field, or equivalent experience"],
        preferred_qualifications=["Experience with Kubernetes", "Scripting proficiency in Python or Bash"],
        stand_out=["Experience running large-scale production infrastructure"],
        benefits=["Stock unit awards (RSUs)", "Comprehensive health coverage", "Relocation assistance"],
    ),
    dict(
        job_title="UX Designer, Customer Experience",
        job_category="Design",
        experience="2-3 years",
        location="Seattle, WA",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="$110,000 - $140,000 / year",
        required_skills=["Figma", "User Research", "Interaction Design", "Prototyping"],
        preferred_skills=["Design Systems", "Accessibility (WCAG)", "Usability Testing"],
        responsibilities=[
            "Design intuitive, accessible experiences for Amazon's customer-facing products",
            "Conduct and synthesize user research to inform design decisions",
            "Partner with product and engineering to ship high-quality experiences",
        ],
        job_summary=(
            "We're looking for a UX Designer to craft intuitive, delightful experiences for "
            "customers shopping on Amazon, from discovery through checkout."
        ),
        about_role=(
            "You'll partner with product managers, researchers, and engineers throughout the "
            "design process — from early concept sketches through high-fidelity prototypes and "
            "production handoff. We value designers who ground their decisions in user research "
            "and data."
        ),
        major_accountabilities=[
            "Create wireframes, prototypes, and high-fidelity designs for customer-facing features",
            "Conduct user research and usability testing to validate design decisions",
            "Collaborate with engineering to ensure design fidelity in production",
            "Contribute to and maintain shared design system components",
        ],
        minimum_requirements=["2-3 years of UX/product design experience", "Strong portfolio demonstrating end-to-end design process", "Proficiency with Figma"],
        required_qualifications=["Bachelor's degree in Design, HCI, or related field, or equivalent practical experience"],
        preferred_qualifications=["Experience contributing to a design system", "Familiarity with WCAG accessibility standards"],
        stand_out=["Experience designing for large-scale e-commerce products"],
        benefits=["Stock unit awards (RSUs)", "Comprehensive health coverage", "Career development programs"],
    ),
    dict(
        job_title="Machine Learning Engineer, AWS AI",
        job_category="AI / Machine Learning",
        experience="4-6 years",
        location="Bangalore",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Master's degree",
        salary="₹35,00,000 - ₹50,00,000 / year",
        required_skills=["Python", "Deep Learning", "PyTorch", "Distributed Training"],
        preferred_skills=["MLOps", "Kubernetes", "Large Language Models"],
        responsibilities=[
            "Build and optimize large-scale machine learning training pipelines",
            "Deploy models into production AWS AI services",
            "Collaborate with research teams to productionize new model architectures",
        ],
        job_summary=(
            "AWS AI is looking for a Machine Learning Engineer to build and scale the training and "
            "deployment infrastructure behind our AI services, used by customers around the world."
        ),
        about_role=(
            "You'll work at the intersection of research and engineering, taking promising model "
            "architectures and making them production-ready — fast, reliable, and cost-efficient "
            "at scale. This role requires strong software engineering fundamentals alongside deep "
            "ML expertise."
        ),
        major_accountabilities=[
            "Design and implement distributed training pipelines for large models",
            "Optimize model inference for latency, throughput, and cost",
            "Partner with research scientists to productionize new architectures",
            "Build tooling and infrastructure to support the ML lifecycle (MLOps)",
        ],
        minimum_requirements=["4-6 years of experience in machine learning engineering", "Strong Python and PyTorch skills", "Experience with distributed training"],
        required_qualifications=["Master's degree in Computer Science, Machine Learning, or related field"],
        preferred_qualifications=["Experience with large language models", "Familiarity with Kubernetes-based ML infrastructure"],
        stand_out=["Publications or open-source contributions in ML systems"],
        benefits=["Stock unit awards (RSUs)", "Comprehensive health coverage", "Career development programs"],
    ),
    dict(
        job_title="Warehouse Operations Manager",
        job_category="Operations",
        experience="2-3 years",
        location="Chennai",
        work_mode="Onsite",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="₹12,00,000 - ₹18,00,000 / year",
        required_skills=["Team Leadership", "Inventory Management", "Process Improvement"],
        preferred_skills=["Lean/Six Sigma", "WMS Software", "Safety Compliance"],
        responsibilities=[
            "Lead a team of associates across daily fulfillment center operations",
            "Ensure safety, quality, and productivity targets are consistently met",
            "Drive continuous improvement initiatives on the warehouse floor",
        ],
        job_summary=(
            "We're hiring a Warehouse Operations Manager to lead day-to-day operations at one of "
            "our fulfillment centers, ensuring safety, quality, and efficiency across a large "
            "team."
        ),
        about_role=(
            "You'll manage a team of associates and shift supervisors, own daily operational "
            "performance, and drive process improvements to increase throughput while maintaining "
            "a strong safety culture. This is a hands-on, floor-based leadership role."
        ),
        major_accountabilities=[
            "Lead and develop a team of associates and shift supervisors",
            "Monitor daily performance metrics and address gaps in real time",
            "Champion a strong safety culture and ensure compliance with protocols",
            "Identify and implement process improvements to increase efficiency",
        ],
        minimum_requirements=["2-3 years of experience in operations or team leadership", "Comfort working in a fast-paced, physical environment"],
        required_qualifications=["Bachelor's degree or equivalent operational leadership experience"],
        preferred_qualifications=["Lean or Six Sigma exposure", "Experience with warehouse management systems (WMS)"],
        stand_out=["Experience managing large teams (50+) in a logistics environment"],
        benefits=["Stock unit awards (RSUs)", "Comprehensive health coverage", "Career development programs"],
    ),
    dict(
        job_title="Technical Product Manager",
        job_category="Product Management",
        experience="4-6 years",
        location="Worldwide",
        work_mode="Remote",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="$135,000 - $175,000 / year",
        required_skills=["Product Management", "Technical Roadmapping", "Stakeholder Management", "SQL"],
        preferred_skills=["API Design", "Agile/Scrum", "Data Analysis"],
        responsibilities=[
            "Own the product roadmap for a core platform capability",
            "Translate customer and business needs into clear technical requirements",
            "Partner closely with engineering to ship and iterate on features",
        ],
        job_summary=(
            "We're looking for a Technical Product Manager to own the roadmap for a core platform "
            "capability, working closely with engineering to turn ambiguous problems into shipped "
            "solutions."
        ),
        about_role=(
            "You'll define product strategy, write clear technical requirements, and partner "
            "day-to-day with engineering teams to prioritize and deliver. This role requires "
            "enough technical depth to have credible conversations with engineers about "
            "trade-offs and architecture."
        ),
        major_accountabilities=[
            "Define and communicate a clear product roadmap and prioritization",
            "Write detailed requirements and acceptance criteria for engineering",
            "Analyze usage data to inform product decisions",
            "Coordinate launches across engineering, design, and business stakeholders",
        ],
        minimum_requirements=["4-6 years of product management experience", "Comfort working directly with engineering on technical trade-offs", "Strong written and verbal communication"],
        required_qualifications=["Bachelor's degree in a technical field or equivalent experience"],
        preferred_qualifications=["Experience with API-based platform products", "Working SQL proficiency for data-informed decisions"],
        stand_out=["Prior experience as a software engineer before moving into product"],
        benefits=["Stock unit awards (RSUs)", "Fully remote flexibility", "Comprehensive health coverage"],
    ),
]

GOOGLE_JOBS = [
    dict(
        job_title="Software Engineer, Backend",
        job_category="Software Engineering",
        experience="2-3 years",
        location="Bangalore",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="₹30,00,000 - ₹45,00,000 / year",
        required_skills=["Go", "Distributed Systems", "gRPC", "Data Structures & Algorithms"],
        preferred_skills=["Kubernetes", "Bigtable", "C++"],
        responsibilities=[
            "Design and build backend services for a high-traffic Google product",
            "Improve system reliability, latency, and scalability",
            "Collaborate with cross-functional teams across the stack",
        ],
        job_summary="We're hiring a Backend Software Engineer to design and build highly reliable, low-latency services that operate at Google scale.",
        about_role=(
            "You'll join a product engineering team responsible for services used by millions of "
            "users daily. Expect a strong emphasis on code quality, testing, and system design, "
            "alongside real ownership of what you build."
        ),
        major_accountabilities=[
            "Design and implement backend services with strong reliability guarantees",
            "Write comprehensive tests and participate in code reviews",
            "Debug and resolve production issues across the stack",
            "Contribute to system design discussions and technical documentation",
        ],
        minimum_requirements=["2-3 years of professional software engineering experience", "Strong CS fundamentals (data structures, algorithms)", "Experience with at least one strongly-typed language"],
        required_qualifications=["Bachelor's degree in Computer Science or equivalent practical experience"],
        preferred_qualifications=["Experience with Go", "Familiarity with distributed storage systems"],
        stand_out=["Contributions to large-scale open-source projects"],
        benefits=["Equity compensation", "Comprehensive health coverage", "Learning & development stipend"],
    ),
    dict(
        job_title="Cloud Solutions Architect",
        job_category="Cloud / Infrastructure",
        experience="7+ years",
        location="Worldwide",
        work_mode="Remote",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="$155,000 - $200,000 / year",
        required_skills=["Google Cloud Platform", "Kubernetes", "Networking", "Enterprise Architecture"],
        preferred_skills=["Terraform", "Security & Compliance", "Customer-Facing Consulting"],
        responsibilities=[
            "Design cloud architectures for enterprise customers migrating to GCP",
            "Act as a trusted technical advisor across the customer lifecycle",
            "Deliver architecture workshops and best-practice guidance",
        ],
        job_summary="Google Cloud is looking for a Solutions Architect to guide enterprise customers through designing and adopting scalable, secure cloud architectures on GCP.",
        about_role=(
            "You'll be the lead technical advisor for a portfolio of strategic accounts, helping "
            "translate business goals into concrete cloud architecture recommendations, and "
            "supporting customers through migration and ongoing optimization."
        ),
        major_accountabilities=[
            "Design secure, scalable GCP architectures tailored to customer needs",
            "Lead technical workshops and proof-of-concept engagements",
            "Build long-term relationships with customer technical leadership",
            "Stay current with new GCP services and industry trends",
        ],
        minimum_requirements=["7+ years in cloud architecture, infrastructure, or consulting", "Deep hands-on GCP experience", "Excellent customer-facing communication skills"],
        required_qualifications=["Bachelor's degree in a technical field or equivalent experience"],
        preferred_qualifications=["Google Cloud Professional Architect certification", "Experience with Terraform-based infrastructure automation"],
        stand_out=["Experience leading large enterprise cloud migrations end-to-end"],
        benefits=["Equity compensation", "Fully remote flexibility", "Comprehensive health coverage"],
    ),
    dict(
        job_title="UX Researcher",
        job_category="Design",
        experience="2-3 years",
        location="Mountain View, CA",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Master's degree",
        salary="$120,000 - $155,000 / year",
        required_skills=["User Research", "Qualitative Research", "Survey Design", "Data Synthesis"],
        preferred_skills=["Statistics", "Figma", "Accessibility Research"],
        responsibilities=[
            "Plan and conduct user research studies to inform product decisions",
            "Synthesize findings into clear, actionable recommendations",
            "Partner with designers and PMs throughout the product lifecycle",
        ],
        job_summary="We're looking for a UX Researcher to help teams deeply understand user needs and translate them into better product decisions.",
        about_role=(
            "You'll design and run qualitative and quantitative research studies across the "
            "product lifecycle — from early exploratory research through post-launch evaluation — "
            "and partner closely with designers, PMs, and engineers to act on findings."
        ),
        major_accountabilities=[
            "Design and conduct user research studies (interviews, surveys, usability tests)",
            "Synthesize research findings into clear insights and recommendations",
            "Present findings to cross-functional stakeholders and drive action",
            "Build and maintain research repositories for institutional knowledge",
        ],
        minimum_requirements=["2-3 years of UX research experience", "Strong qualitative and quantitative research skills", "Excellent communication and storytelling ability"],
        required_qualifications=["Master's degree in HCI, Psychology, or related field, or equivalent experience"],
        preferred_qualifications=["Statistical analysis experience", "Familiarity with accessibility research methods"],
        stand_out=["Published research in HCI or related fields"],
        benefits=["Equity compensation", "Comprehensive health coverage", "Learning & development stipend"],
    ),
]

NETFLIX_JOBS = [
    dict(
        job_title="Data Engineer",
        job_category="Data / Analytics",
        experience="4-6 years",
        location="Los Gatos, CA",
        work_mode="Hybrid",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="$160,000 - $210,000 / year",
        required_skills=["Python", "SQL", "Spark", "Data Pipeline Design"],
        preferred_skills=["Airflow", "AWS", "Streaming Data (Kafka/Flink)"],
        responsibilities=[
            "Build and maintain data pipelines powering recommendation and analytics systems",
            "Ensure data quality, reliability, and scalability across pipelines",
            "Partner with data science and analytics teams on data needs",
        ],
        job_summary="Netflix is looking for a Data Engineer to build the pipelines that power our recommendation systems and business analytics at global scale.",
        about_role=(
            "You'll design, build, and operate data pipelines that process massive volumes of "
            "viewing and engagement data, working closely with data scientists and analysts to "
            "ensure the data they depend on is timely, accurate, and well-modeled."
        ),
        major_accountabilities=[
            "Design and build scalable, reliable data pipelines",
            "Monitor data quality and resolve pipeline issues proactively",
            "Partner with data science teams to understand and meet data needs",
            "Continuously improve pipeline performance and cost efficiency",
        ],
        minimum_requirements=["4-6 years of data engineering experience", "Strong Python and SQL skills", "Experience with distributed data processing (Spark or similar)"],
        required_qualifications=["Bachelor's degree in Computer Science or related field, or equivalent experience"],
        preferred_qualifications=["Experience with workflow orchestration tools (Airflow)", "Familiarity with streaming data systems"],
        stand_out=["Experience building data infrastructure at streaming/media scale"],
        benefits=["Unlimited vacation policy", "Top-of-market compensation", "Comprehensive health coverage"],
    ),
    dict(
        job_title="Content Analyst",
        job_category="Data / Analytics",
        experience="2-3 years",
        location="Los Angeles, CA",
        work_mode="Onsite",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="$105,000 - $135,000 / year",
        required_skills=["SQL", "Data Analysis", "Excel", "Data Visualization"],
        preferred_skills=["Python", "Tableau", "Statistics"],
        responsibilities=[
            "Analyze viewership and engagement data to inform content decisions",
            "Build reports and dashboards for content strategy teams",
            "Partner with content and marketing stakeholders on analysis requests",
        ],
        job_summary="We're hiring a Content Analyst to turn viewership data into insights that inform content investment and programming decisions.",
        about_role=(
            "You'll work closely with content strategy teams to answer questions about audience "
            "behavior, title performance, and engagement trends, presenting clear, actionable "
            "analysis to inform high-stakes content decisions."
        ),
        major_accountabilities=[
            "Analyze viewership and engagement trends across titles and genres",
            "Build and maintain dashboards for content performance tracking",
            "Present findings and recommendations to content stakeholders",
            "Support ad-hoc analysis requests from content and marketing teams",
        ],
        minimum_requirements=["2-3 years of analytics experience", "Strong SQL and Excel skills", "Ability to communicate data insights clearly to non-technical audiences"],
        required_qualifications=["Bachelor's degree in a quantitative or business field"],
        preferred_qualifications=["Experience with Tableau or similar BI tools", "Working knowledge of Python for analysis"],
        stand_out=["Experience in media, entertainment, or streaming analytics"],
        benefits=["Unlimited vacation policy", "Top-of-market compensation", "Comprehensive health coverage"],
    ),
    dict(
        job_title="Site Reliability Engineer",
        job_category="Cloud / Infrastructure",
        experience="4-6 years",
        location="Worldwide",
        work_mode="Remote",
        employment_type="Full-time",
        education="Bachelor's degree",
        salary="$170,000 - $220,000 / year",
        required_skills=["AWS", "Kubernetes", "Python", "Incident Response"],
        preferred_skills=["Chaos Engineering", "Observability Tooling", "Go"],
        responsibilities=[
            "Ensure the reliability and performance of Netflix's global streaming infrastructure",
            "Build tooling to improve observability and incident response",
            "Participate in on-call rotation for critical production systems",
        ],
        job_summary="We're looking for a Site Reliability Engineer to help keep Netflix's global streaming infrastructure fast, resilient, and highly available.",
        about_role=(
            "You'll work on the systems and tooling that keep streaming reliable for hundreds of "
            "millions of members, focusing on observability, automated remediation, and proactive "
            "resilience engineering."
        ),
        major_accountabilities=[
            "Improve reliability and performance of critical production systems",
            "Build tooling for observability, alerting, and automated incident response",
            "Participate in on-call rotation and lead post-incident reviews",
            "Drive chaos engineering practices to proactively surface weaknesses",
        ],
        minimum_requirements=["4-6 years of SRE, infrastructure, or backend engineering experience", "Strong AWS and Kubernetes experience", "Solid Python or Go skills"],
        required_qualifications=["Bachelor's degree in Computer Science or related field, or equivalent experience"],
        preferred_qualifications=["Experience with chaos engineering practices", "Familiarity with modern observability stacks"],
        stand_out=["Experience operating infrastructure at global, multi-region scale"],
        benefits=["Unlimited vacation policy", "Top-of-market compensation", "Fully remote flexibility"],
    ),
]

JOBS_BY_COMPANY = {
    "Amazon": AMAZON_JOBS,
    "Google": GOOGLE_JOBS,
    "Netflix": NETFLIX_JOBS,
}


def publish_job(company_id: int, company_name: str, owner_user_id: int, job: dict, company_overview: str, why_join_us: str) -> str:
    session_id = str(uuid.uuid4())
    job_state = {
        "job_title": job["job_title"],
        "job_category": job["job_category"],
        "experience": job["experience"],
        "location": job["location"],
        "work_mode": job["work_mode"],
        "employment_type": job["employment_type"],
        "education": job["education"],
        "salary": job["salary"],
        "required_skills": job["required_skills"],
        "preferred_skills": job["preferred_skills"],
        "responsibilities": job["responsibilities"],
    }
    upsert_job_draft(session_id, company_id, job_state, jd_stale=False, owner_user_id=owner_user_id)

    jd = {
        "job_title": job["job_title"],
        "requisition_id": None,
        "job_category": job["job_category"],
        "employment_type": job["employment_type"],
        "location": job["location"],
        "work_mode": job["work_mode"],
        "deadline": None,
        "company_overview": company_overview,
        "job_summary": job["job_summary"],
        "about_role": job["about_role"],
        "major_accountabilities": job["major_accountabilities"],
        "minimum_requirements": job["minimum_requirements"],
        "required_qualifications": job["required_qualifications"],
        "preferred_qualifications": job["preferred_qualifications"],
        "required_skills": job["required_skills"],
        "stand_out": job["stand_out"],
        "benefits": job["benefits"],
        "why_company": why_join_us,
    }
    jd_versions = {"1": jd}
    save_jd_versions(session_id, jd_versions)
    save_selected_version(session_id, "1")

    result = finalize_publish(session_id, company_id, company_name, jd, "1")
    return result["job_id"]


def seed() -> None:
    """Idempotent: silently skips any company whose name/recruiter email already exists, so this
    is also safe to call from main.py's startup hook (see BOOTSTRAP_DEMO_JOBS in config.py) —
    the same "no DB shell access on Railway" workaround already used for BOOTSTRAP_ADMIN_*.
    """
    init_db()
    for entry in COMPANIES:
        try:
            created = create_company_and_recruiter(
                company_name=entry["company_name"],
                company_fields=entry["profile"],
                email=entry["recruiter_email"],
                password_hash=hash_password(entry["recruiter_password"]),
                name=entry["recruiter_name"],
            )
        except Exception as exc:  # sqlite3.IntegrityError on a re-run
            print(f"Skipping {entry['company_name']} — likely already exists ({exc})")
            continue

        company_id = created["company"]["id"]
        print(f"Created company={entry['company_name']!r} id={company_id} "
              f"recruiter={entry['recruiter_email']!r}")

        jobs = JOBS_BY_COMPANY[entry["company_name"]]
        overview = entry["profile"]["company_overview"]
        why_join = entry["profile"]["why_join_us"]
        for job in jobs:
            job_id = publish_job(company_id, entry["company_name"], created["user"]["id"], job, overview, why_join)
            print(f"  Published {job_id}: {job['job_title']}")


if __name__ == "__main__":
    seed()
