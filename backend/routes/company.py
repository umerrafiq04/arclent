from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_recruiter
from backend.database import get_company_profile_by_id, update_company_profile
from backend.schemas import CompanyProfileUpdate

router = APIRouter(prefix="/api/company-profile", tags=["company"])


@router.get("")
def get_company_profile(user: dict = Depends(get_current_recruiter)) -> dict:
    profile = get_company_profile_by_id(user["company_id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Company profile not found")
    return profile


@router.put("")
def put_company_profile(body: CompanyProfileUpdate, user: dict = Depends(get_current_recruiter)) -> dict:
    # company_id always comes from the authenticated session, never the request body —
    # there is nothing here for a client to tamper with to reach another company's row.
    updated = update_company_profile(user["company_id"], body.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update company profile")
    return updated
