import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from backend.auth import (
    COOKIE_NAME,
    check_signin_rate_limit,
    clear_session_cookie,
    clear_signin_attempts,
    create_session,
    get_current_user,
    hash_password,
    record_signin_attempt,
    set_session_cookie,
    verify_login,
)
from backend.database import (
    create_company_and_recruiter,
    delete_auth_session,
    get_company_profile_by_id,
    get_company_profile_by_name,
    get_user_by_email,
)
from backend.schemas import SigninRequest, SignupRequest, UserResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])

MIN_PASSWORD_LENGTH = 8


def _user_response(user: dict) -> UserResponse:
    company_name = None
    if user.get("company_id") is not None:
        profile = get_company_profile_by_id(user["company_id"])
        company_name = profile["company_name"] if profile else None
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
        company_id=user.get("company_id"),
        company_name=company_name,
    )


@router.post("/signup", response_model=UserResponse, status_code=201)
def signup(body: SignupRequest, response: Response) -> UserResponse:
    email = body.email.strip().lower()
    name = body.name.strip()
    company_name = body.company_name.strip()

    if len(body.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=422, detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")
    if not company_name:
        raise HTTPException(status_code=422, detail="Company name is required.")

    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="That email is already registered.")
    if get_company_profile_by_name(company_name):
        raise HTTPException(status_code=409, detail="That company name is already registered.")

    company_fields = body.model_dump(exclude={"name", "email", "password", "company_name"}, exclude_unset=True)
    password_hash = hash_password(body.password)
    try:
        created = create_company_and_recruiter(company_name, company_fields, email, password_hash, name)
    except sqlite3.IntegrityError:
        # Race with another signup for the same email/company_name between the checks above
        # and the insert — rare, but must still surface as a clean error, not a 500.
        raise HTTPException(status_code=409, detail="That email or company name is already registered.")

    user = created["user"]
    token = create_session(user["id"])
    set_session_cookie(response, token)
    return _user_response(user)


@router.post("/signin", response_model=UserResponse)
def signin(body: SigninRequest, response: Response) -> UserResponse:
    email = body.email.strip().lower()
    check_signin_rate_limit(email)

    user = verify_login(email, body.password)
    if user is None:
        record_signin_attempt(email)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    clear_signin_attempts(email)
    token = create_session(user["id"])
    set_session_cookie(response, token)
    return _user_response(user)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        delete_auth_session(token)
    clear_session_cookie(response)


@router.get("/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)) -> UserResponse:
    return _user_response(user)
