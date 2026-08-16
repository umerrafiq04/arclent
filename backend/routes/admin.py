from fastapi import APIRouter, Depends, HTTPException

from backend.auth import require_admin
from backend.database import get_admin_job_by_id, get_company_profile_by_id, list_all_jobs_admin, list_jobs_for_company

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/jobs")
def get_admin_jobs(admin: dict = Depends(require_admin)) -> list[dict]:
    """All jobs, all companies, all statuses — Page 4's admin table."""
    return list_all_jobs_admin()


@router.get("/jobs/{job_id}")
def get_admin_job(job_id: int, admin: dict = Depends(require_admin)) -> dict:
    job = get_admin_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/companies/{company_id}")
def get_admin_company(company_id: int, admin: dict = Depends(require_admin)) -> dict:
    """Company profile + every job posted by that company — the admin drilldown view."""
    profile = get_company_profile_by_id(company_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Company not found")
    return {"company": profile, "jobs": list_jobs_for_company(company_id)}
