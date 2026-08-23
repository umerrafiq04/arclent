import csv
import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.auth import get_current_recruiter
from backend.database import (
    create_application,
    delete_job,
    get_job_by_session_id,
    get_published_job_by_job_id,
    list_applications_for_job,
    list_jobs_for_company,
    list_published_jobs,
    set_accepting_applications,
)
from backend.schemas import AcceptingApplicationsUpdate, JobApplicationRequest

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
public_router = APIRouter(prefix="/api/public/jobs", tags=["public-jobs"])


@router.get("")
def get_jobs_for_recruiter(user: dict = Depends(get_current_recruiter)) -> list[dict]:
    """Jobs (draft + published) for the authenticated recruiter's own company only."""
    return list_jobs_for_company(user["company_id"])


@router.get("/report")
def get_jobs_report(user: dict = Depends(get_current_recruiter)) -> Response:
    """CSV export of this company's own jobs — real fields only (title, status, location,
    employment type, posted date, accepting-applications). No proposal/hire counts: there is
    no applications-tracking table in this schema yet, so those numbers don't exist to export.
    """
    jobs = list_jobs_for_company(user["company_id"])
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["Job ID", "Title", "Status", "Employment Type", "Location", "Work Mode", "Posted Date", "Accepting Applications"]
    )
    for job in jobs:
        writer.writerow(
            [
                job.get("job_id") or "",
                job.get("job_title") or "",
                job.get("status") or "",
                job.get("employment_type") or "",
                job.get("location") or "",
                job.get("work_mode") or "",
                job.get("published_at") or job.get("created_at") or "",
                "Yes" if job.get("accepting_applications") else "No",
            ]
        )
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=arclent-jobs-report.csv"},
    )


@router.put("/{session_id}/accepting-applications")
def put_accepting_applications(
    session_id: str,
    body: AcceptingApplicationsUpdate,
    user: dict = Depends(get_current_recruiter),
) -> dict:
    """Closes/reopens a published job to new applicants — the job stays published and stays
    listed publicly either way; this only toggles whether it's still accepting applicants.
    """
    job = get_job_by_session_id(session_id)
    if not job or job["company_id"] != user["company_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "published":
        raise HTTPException(status_code=400, detail="Only a published job can be opened or closed to applications.")
    return set_accepting_applications(session_id, body.accepting_applications)


@router.delete("/{session_id}")
def delete_job_route(session_id: str, user: dict = Depends(get_current_recruiter)) -> dict:
    """Permanently removes a job (draft or published) belonging to the recruiter's own company."""
    job = get_job_by_session_id(session_id)
    if not job or job["company_id"] != user["company_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    delete_job(session_id)
    return {"deleted": True}


@public_router.get("")
def get_public_jobs() -> list[dict]:
    """Published jobs only, across all companies — Page 3's public listing."""
    return list_published_jobs()


@public_router.get("/{job_id}")
def get_public_job(job_id: str) -> dict:
    job = get_published_job_by_job_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@public_router.post("/{job_id}/apply")
def apply_to_job(job_id: str, body: JobApplicationRequest) -> dict:
    """A candidate's submission on the public Apply form — no auth, anyone can apply."""
    job = get_published_job_by_job_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("accepting_applications"):
        raise HTTPException(status_code=400, detail="This job is no longer accepting applications.")
    # Only keep answers for questions that actually exist on this job right now — the recruiter may
    # have edited custom_questions since the candidate loaded the page, so the submitted keys aren't
    # trusted as-is.
    current_questions = set(job.get("custom_questions") or [])
    answers = {q: a for q, a in body.answers.items() if q in current_questions}
    return create_application(job_id, answers)


@router.get("/{session_id}/applications")
def get_applications_for_job(session_id: str, user: dict = Depends(get_current_recruiter)) -> list[dict]:
    """Submitted applications for one of the recruiter's own jobs, newest first."""
    job = get_job_by_session_id(session_id)
    if not job or job["company_id"] != user["company_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("job_id"):
        return []
    return list_applications_for_job(job["job_id"])
