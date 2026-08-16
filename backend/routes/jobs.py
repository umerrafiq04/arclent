from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_recruiter
from backend.database import (
    get_job_by_session_id,
    get_published_job_by_job_id,
    list_jobs_for_company,
    list_published_jobs,
    set_accepting_applications,
)
from backend.schemas import AcceptingApplicationsUpdate

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
public_router = APIRouter(prefix="/api/public/jobs", tags=["public-jobs"])


@router.get("")
def get_jobs_for_recruiter(user: dict = Depends(get_current_recruiter)) -> list[dict]:
    """Jobs (draft + published) for the authenticated recruiter's own company only."""
    return list_jobs_for_company(user["company_id"])


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
